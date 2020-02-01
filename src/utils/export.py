import shutil

# def merge export
from tensorboard.backend.event_processing import event_accumulator

from utils.tools import generate_name

# def merge_exports():


def export_tensorboard_to_csv(source, dest, model_path):
    with open(source) as input:
        configs = input.readlines()
        with open(dest, "w") as output:
            output.write(configs[0][0:-1] + ",last_loss, last_val_loss, last_accuracy, last_val_accuracy\n")
            for conf in configs[1::]:
                filename = generate_name(conf.split(","))
                if filename[-1] == "\n":
                    filename = filename[0:-1] + "_sparse"
                train_path = model_path + filename + "\\train\\"
                val_path = model_path + filename + "\\validation\\"

                ea = event_accumulator.EventAccumulator(path=train_path)
                ea_val = event_accumulator.EventAccumulator(path=val_path)
                ea.Reload()
                ea_val.Reload()
                loss = ea.Scalars('epoch_loss')[-1][2]
                loss_val = ea_val.Scalars('epoch_loss')[-1][2]
                accuracy = ea.Scalars('epoch_sparse_categorical_accuracy')[-1][2]
                accuracy_val = ea_val.Scalars('epoch_sparse_categorical_accuracy')[-1][2]

                if conf[-1] == "\n":
                    conf = conf[0:-1]
                output.write(conf + "," + str(round(loss, 5)) + "," + str(round(loss_val, 5)) + "," + str(round(accuracy, 5)) + "," + str(round(accuracy_val, 5)) + "\n")
