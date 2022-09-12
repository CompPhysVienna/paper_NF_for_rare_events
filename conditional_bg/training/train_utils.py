import numpy as np


def get_epoch_info(s, network):
    
    msg = " Epochs:\t{}".format(s["epochs"])

    msg += " \n LR:\t\t{:.2e}".format(s["LR"])
    msg += "  \n Batch Size:\t{}".format(s["batch_size"])
    
    msg += "  \n N_Conditions:\t{}".format(s["conditions_per_batch"])
        
    if isinstance(s["T"], list):
        msg += " \n T:\t\t{} - {}".format(s["T"][0],s["T"][1])
    else:
        msg += " \n T:\t\t{}".format(s["T"])
        
    msg += " \n w_xz/w_zx:\t{}/{}".format(s["w_xz"],s["w_zx"])
    
    if network._angle_loss:
        msg += " \n w_angle:\t{:.2e}".format(s["w_angle"])
        
    if s["clamp_val"] is not None:
        msg += " \n Clamp:\t\t{:.2e}".format(s["clamp_val"])

    msg = "\n" + msg + "\n"
    
    return msg
    
    
def train_progress_message(s, network, epoch=None, epoch_loss=None, epoch_loss_val=None) :
    

    if epoch_loss is not None:
        epoch_loss = np.array(epoch_loss)

    if epoch_loss_val is not None:
        epoch_loss_val = np.array(epoch_loss_val)

    msg =  ""

    if epoch is not None:
        msg += "E: {:3}/{}".format(epoch+1, s["epochs"])

    #
    # Train
    #

    if epoch_loss is not None:

        if epoch is not None:
            msg += "  |  "
        else:
            msg += "\t"

        msg += "Train:   xz {: >6.3f}".format(np.round(epoch_loss[:,0].mean() ,3))

        if s["w_zx"] > 0:
            msg += "  zx {: >10.3e}".format(np.round(epoch_loss[:,1].mean() ,3))

        if s["w_zx"] > 0:
            if network._angle_loss and s["w_angle"] > 0:
                msg += "  A_zx {: >6.3f}".format(np.round(epoch_loss[:,2].mean() ,3))

    #
    # Validation
    #

    if epoch_loss_val is not None:

        if epoch is not None or epoch_loss is not None:
            msg += "  |  "
        else:
            msg += "\t"

        msg += "Val.:   xz {: >6.3f}  zx {: >10.3e}".format(np.round(epoch_loss_val[0] ,3),
                                                                 np.round(epoch_loss_val[1], 3))


        if network._angle_loss:
            msg += "  A_zx {: >6.3f}".format(np.round(epoch_loss_val[2] ,3))
            
        if epoch_loss_val[8] is not None:
            msg += "  F_xz {: >.2f}".format(epoch_loss_val[8])

        if epoch_loss_val[9] is not None:
            msg += "  F_zx {: >.2f}".format(epoch_loss_val[9])
            
        if epoch_loss_val[10] is not None:
            msg += "  E_xz {: >9.2e}".format(epoch_loss_val[10])

        if epoch_loss_val[11] is not None:
            msg += "  E_zx {: >9.2e}".format(epoch_loss_val[11])
        

    return msg
        
        
def check_schedule(schedule, network):
   

    # If angle loss terms are included, the corresponding weight has to be defined
    if network._angle_loss:
        defined_angle_weight = ["w_angle" in s for s in schedule]

        if not all(defined_angle_weight):
            raise Exception("Angle loss weight \"w_angle\" has to be defined for all training schedules when transformations with angle loss terms are used.") 

    for s in schedule:

        assert s["batch_size"] % s["conditions_per_batch"] == 0, "Batch size must be divisible by conditions per batch!"

        if "T_train" in s and not network.prior.multi_T_supported:
            raise Exception("Training at multiple temperatures not supported with the choosen prior model!") 