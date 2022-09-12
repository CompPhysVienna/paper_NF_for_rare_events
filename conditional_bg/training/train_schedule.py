import numpy as np
from scipy.interpolate import CubicSpline

from datetime import datetime

import torch
from torch.utils.data import DataLoader

from conditional_bg import util
from conditional_bg.training import train_utils


class schedule_trainer(object):
    
    def __init__(self, network, reference_configurations=None, reference_conditions=None, latent_configurations=None,
                 energy_test_data = None):
        
        self.network = network
        
        #
        # Low energy Fraction in configuration space
        #
        
        self.energy_test_data = energy_test_data
        
        self._energy_fraction_zx = False
        
        if reference_configurations is not None:
            self.low_energy_function_zx, self.high_energy_function_zx = self.get_energy_functions(reference_configurations, reference_conditions)
            self._energy_fraction_zx = True
        
        #
        # Low energy Fraction in latent space
        #
        
        self._energy_fraction_xz = False
        if latent_configurations is not None:
            
            # Get energies
            with torch.no_grad():
                latent_energies = self.network.prior.log_prob(latent_configurations).cpu().numpy()
                
            self.low_energy_threshold_xz = np.percentile(latent_energies, 99)
            self.high_energy_threshold_xz = np.percentile(latent_energies, 1)
            self._energy_fraction_xz = True
            
        self.device = network.device
        
        
    def get_energy_functions(self, reference_configurations, reference_conditions):

        # Get conditions
        unique_conditions = torch.unique(reference_conditions, sorted=True).cpu().numpy()

        low_percentiles = []
        high_percentiles = []

        for cond in unique_conditions:

            # Get configuations corresponding to certain condition
            cond_configurations = reference_configurations[torch.where(reference_conditions == cond)[0]]

            # Get energies of configuations corresponding to certain condition
            with torch.no_grad():
                
                generated_condition = self.network.condition_function(cond_configurations)
                bias_energies = self.network.likelihood.log_prob(generated_condition, torch.ones_like(generated_condition) * cond)
                
                full_energies = self.network.posterior.log_prob(cond_configurations) + bias_energies

            low_percentiles.append(np.percentile(full_energies.cpu().numpy(), 99))
            high_percentiles.append(np.percentile(full_energies.cpu().numpy(), 1))

        # Interpolate percentiles as function of condition
        low_energy_F = CubicSpline(unique_conditions, low_percentiles)
        high_energy_F = CubicSpline(unique_conditions, high_percentiles)

        return low_energy_F, high_energy_F
    
    
    def train_epoch(self, s, train_dataloader, optimizer, train_scheduler=None):
        
        multi_T_train = False
        if isinstance(s["T"], (tuple, list, np.ndarray)):
            multi_T_train = True
                    
        epoch_loss = []
        
        for i, sample in enumerate(train_dataloader):

            #
            # ML
            #
            
            # Get data samples from dataset
            x = sample[0]
            
            # Get temperatures of samples from dataset
            T_sample = sample[1]
            
            # Get condition of samples from dataset, will be None for unconditioned samples
            xz_condition = sample[2]
            loss_xz, loss_angle_sym, _, _, _ = self.network.loss("xz", x, T = T_sample, condition=sample[2],  clamp_max = s["clamp_val"])
            loss_xz = util.grouped_mean(loss_xz, xz_condition).mean()
           
            #
            # KL
            #
            
            loss_zx, loss_angle = None, None
            
            if s["w_zx"] > 0:
                
                T_prior = None
                
                # Get Temperatures for sampling
                if multi_T_train:

                        T_sample = torch.rand([s["batch_size"], 1], device=self.device, dtype=torch.float32)
                        T_sample *= (s["T"][1]-s["T"][0]) 
                        T_sample += s["T"][0]
                        
                        T_prior = T_sample
                else:

                    if not self.network.prior.multi_T_supported:
                        T_sample = torch.ones(s["batch_size"], 1, device=self.device, dtype=torch.float32) *  s["T"]
                        T_prior = s["T"]
                    else:
                        T_sample = torch.ones(s["batch_size"], 1, device=self.device, dtype=torch.float32) *  s["T"]
                        T_prior = T_sample
                

                ref_condition = torch.rand([s["conditions_per_batch"], self.network.condition_dim], device=self.device, dtype=torch.float32) 
                ref_condition *= (self.network.norm_range_condition[1]-self.network.norm_range_condition[0])
                ref_condition += self.network.norm_range_condition[0]
                
                ref_condition = ref_condition.repeat(s["batch_size"]//s["conditions_per_batch"], 1)
                
                z = self.network.prior.sample(s["batch_size"], T = T_prior)
                
                loss_zx, loss_angle, _, _, _ = self.network.loss("zx", z, T = T_sample, condition=ref_condition, clamp_max=s["clamp_val"])
                
                
                if self.network._conditioned:
                    loss_zx = util.grouped_mean(loss_zx, ref_condition).mean()
                else:
                    loss_zx = loss_zx.mean()


            #
            # Backpropagation
            #

            optimizer.zero_grad()

            loss = loss_xz

            if s["w_zx"] > 0:
                loss = s["w_xz"] * loss_xz + s["w_zx"] * loss_zx

                if self.network._angle_loss and s["w_angle"] > 0:
                    loss +=  s["w_angle"] * loss_angle

            loss.backward()

            optimizer.step()
            
            if train_scheduler is not None:
                train_scheduler.step()
            
            #
            # Logging Train Loss
            #
            
            loss_list = [loss_xz, loss_zx, loss_angle, loss_angle_sym, None, None, None, None]

            loss_list = [l.item() if l is not None else l for l in loss_list]

            epoch_loss.append(loss_list)
                    
        return epoch_loss        
        
        
    def get_energy_fraction(self, s, train_dataset, log_prob_xz, log_prob, condition):
        
        #
        # Low energy fraction
        #
        
        if self.energy_test_data is not None:
            
            with torch.no_grad():

                test_data_xz, test_T_xz, test_c_xz, test_data_zx, test_T_zx, test_c_zx = self.energy_test_data

                # ML

                *_, log_prob_xz = self.network.loss("xz", test_data_xz, T=test_T_xz, 
                                                            condition=test_c_xz, clamp_max=s["clamp_val"])

                # KL
                
                *_, log_prob = self.network.loss("zx", test_data_zx, T=test_T_zx, condition=test_c_zx, 
                                                        clamp_max=s["clamp_val"])
                
                condition = test_c_zx
                    
        # data
        energy_fraction_zx = None
        if self._energy_fraction_zx:

            log_prob_np = log_prob.cpu().numpy()

            threshold_condition = None if condition is None else condition.cpu().numpy()
            low_threshold = self.low_energy_function_zx(threshold_condition)
            high_threshold = self.high_energy_function_zx(threshold_condition)

            energy_fraction_zx = np.count_nonzero( ((log_prob_np - low_threshold) < 0) & \
                                                    ((log_prob_np - high_threshold) > 0) ) / log_prob.shape[0]

        energy_indicator_zx = torch.median(log_prob).item()

            
        # latent
        energy_fraction_xz = None
        if self._energy_fraction_xz:

            log_prob_np = log_prob_xz.cpu().numpy()
            energy_fraction_xz =  np.count_nonzero( ((log_prob_np - self.low_energy_threshold_xz) < 0) & \
                                                    ((log_prob_np - self.high_energy_threshold_xz) > 0) ) / log_prob.shape[0]

        energy_indicator_xz = torch.median(log_prob_xz).item()

            
        return [energy_fraction_xz, energy_fraction_zx, energy_indicator_xz, energy_indicator_zx]
        
        
        
    def test_model(self, s, train_dataset):
        
        
        with torch.no_grad():
            
            test_data_xz, test_T_xz, test_c_xz, test_data_zx, test_T_zx, test_c_zx = train_dataset.get_test_data()

            #
            # ML
            #
           
            loss_xz, loss_angle_xz, _, _, log_prob_xz = self.network.loss("xz", test_data_xz, T=test_T_xz, 
                                                                                                condition=test_c_xz, clamp_max=s["clamp_val"])
            loss_xz = util.grouped_mean(loss_xz, test_c_xz).mean()
           

            #
            # KL
            #
            
            loss_zx, loss_angle, _, _, log_prob = self.network.loss("zx", test_data_zx, T=test_T_zx, condition=test_c_zx, 
                                                                    clamp_max=s["clamp_val"])

            loss_zx = util.grouped_mean(loss_zx, test_c_zx).mean()
               
            epoch_loss_val = [loss_xz, loss_zx, loss_angle, loss_angle_xz, None, None, None, None]

            epoch_loss_val = [l.item() if l is not None else l for l in epoch_loss_val]

            epoch_loss_val += self.get_energy_fraction(s, train_dataset, log_prob_xz, log_prob, test_c_zx)
        
        return epoch_loss_val
    
    
    def train_schedule(self, schedule_file, train_dataset, checkpoint_dir=None, chk_frequency=25,
                       optimizer=None, optimizer_args=None, 
                       scheduler=None, scheduler_args=None, rerun_history=None, verbose=True):
           
            schedule = util.load_yaml(schedule_file)
            
            
            if rerun_history is not None:
                
                loss_history = np.load(rerun_history, allow_pickle=True)
                self.rerun_training_dry(schedule, loss_history)
                
                return loss_history
                
            
            # Perform checks that the schedule and initialized model are compatible
            train_utils.check_schedule(schedule, self.network)

            opt_parameters = [p for p in self.network.realNVP_blocks.parameters() if p.requires_grad==True]

            if optimizer is not None:
                train_optimizer = optimizer(opt_parameters, **optimizer_args)
            else:
                train_optimizer = torch.optim.Adam(opt_parameters, lr=1e-4)
            
            train_scheduler = None
            if scheduler is not None:
                train_scheduler = scheduler(train_optimizer, **scheduler_args)

            _checkpoint = False
            if checkpoint_dir is not None:
                _checkpoint = True
                filename = checkpoint_dir + "/network_{}.{}.pt"
                
                util.write_yaml(checkpoint_dir + "/train_protocol.yaml", schedule)

                
            #
            # TRAINING
            #

            total_epochs = 0
            training_loss = []

            try:
                
                if _checkpoint:
                    
                    with open(checkpoint_dir + "/train_log.txt", "w") as file:

                        msg = str(self.network) + "\n\n\n"
                        
                        msg += "--------------------------------------------------------------\n"
                        msg += "                      Training Summary                        \n"
                        msg += "--------------------------------------------------------------\n\n"
        
                        start = datetime.now()
                        file.write(msg + "Training started: " + start.strftime("%d/%m/%Y, %H:%M:%S") + "\n")
                        

                for s_index, s in enumerate(schedule):

                    # Print info, set LR and create dataloader
                    if verbose:
                        
                        epoch_info = train_utils.get_epoch_info(s, self.network)
                        print(epoch_info)

                        if _checkpoint:
                            with open(checkpoint_dir + "/train_log.txt", "a") as file:
                                file.write("\n" + epoch_info + "\n")
                        

                    for param_group in train_optimizer.param_groups:
                            param_group['lr'] = s["LR"]

                    train_dataloader= DataLoader(train_dataset, batch_size=s["batch_size"],
                                            shuffle=True, num_workers=0)

                    
                    for epoch in range(s["epochs"]):
        
                        self.network.train()
                        epoch_loss = self.train_epoch(s, train_dataloader, train_optimizer, train_scheduler)

                        #
                        # Logging/Printing Train/Validation Loss
                        #

                        if verbose:

                            self.network.eval()
                            epoch_loss_val = self.test_model(s, train_dataset)

                            total_epochs += 1
                            training_loss.append([total_epochs, epoch_loss_val, np.array(epoch_loss)])

                            msg = train_utils.train_progress_message(s, self.network, epoch=epoch, 
                                                                     epoch_loss=epoch_loss, epoch_loss_val=epoch_loss_val)
                            
                            print(msg)

                            if _checkpoint:
                                with open(checkpoint_dir + "/train_log.txt", "a") as file:
                                    file.write(msg + "\n")

                        if _checkpoint and (epoch + 1) % chk_frequency == 0:
                            torch.save(self.network.state_dict(), filename.format(s_index, epoch + 1))
                            

                    if _checkpoint:
                        torch.save(self.network.state_dict(), filename.format(s_index, "F"))

                        with open(checkpoint_dir + "/train_log.txt", "a") as file:
                            intermediate = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
                            file.write("\nSchedule {} finished: ".format(s_index) + intermediate + "\n")
                        
                    
            except KeyboardInterrupt:

                if _checkpoint:
                    np.save(checkpoint_dir + "/loss_history.npy", np.array(training_loss, dtype=object))
                    torch.save(self.network.state_dict(), (checkpoint_dir + "/network.pt"))

                return training_loss

            
            if _checkpoint:
                np.save(checkpoint_dir + "/loss_history.npy", np.array(training_loss, dtype=object))
                torch.save(self.network.state_dict(), (checkpoint_dir + "/network.pt"))
            
                with open(checkpoint_dir + "/train_log.txt", "a") as file:
                    end = datetime.now()
                    file.write("\n\nTraining finished: " + end.strftime("%d/%m/%Y, %H:%M:%S"))
                    duration = str(end-start)
                    file.write("\nTime spend training: " + duration)
            
            return training_loss

            
    def rerun_training_dry(self, schedule, loss_history):
        
        rerun_history_test = np.array([loss_history[i][1] for i in range(len(loss_history))])
        rerun_history_train = np.array([loss_history[i][2] for i in range(len(loss_history))], dtype=object)
                                       
        total_epochs = 0
        training_loss = []

        for s_index, s in enumerate(schedule):

            epoch_info = train_utils.get_epoch_info(s, self.network)
            print(epoch_info)

            for epoch in range(s["epochs"]):

                epoch_loss = rerun_history_train[total_epochs]

                epoch_loss_val = rerun_history_test[total_epochs]

                total_epochs += 1

                msg = train_utils.train_progress_message(s, self.network, epoch=epoch, 
                                                         epoch_loss=epoch_loss, epoch_loss_val=epoch_loss_val)
                
                print(msg)