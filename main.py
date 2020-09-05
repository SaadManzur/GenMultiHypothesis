from datasets.SurrealDataset import SurrealDataset
import os
import time
import argparse
import numpy as np
from tensorflow.core.framework.summary_pb2 import SummaryMetadata
from tqdm  import tqdm
import tensorflow as tf
import src.cameras as cameras
from tensorflow.python.client import device_lib
from progress.bar import Bar
from src.mix_den_model import LinearModel
from datasets.utils import AverageMeter, unnormalize_data
from datasets.visualizer import plot_3d

from datasets.GPADataset import GPADataset
from datasets.H36MDataset import H36MDataset
from datasets.ThreeDPWDataset import ThreeDPWDataset

import logging, logging.config
from six.moves import xrange

FLAGS = {}
FLAGS["learning_rate"] = 1e-3 # "Learning rate"
FLAGS["dropout"] = 0.5 # "Dropout keep probability. 1 means no dropout"
FLAGS["batch_size"] = 64 #"batch size to use during training")
FLAGS["epochs"] = 200 # "How many epochs we should train for")
FLAGS["camera_frame"] = True # "Convert 3d poses to camera coordinates")
FLAGS["max_norm"] = True # "Apply maxnorm constraint to the weights")
FLAGS["batch_norm"] = True # "Use batch_normalization")

# Data loading
FLAGS["predict_14"] = False # "predict 14 joints")
FLAGS["use_sh"] = True # "Use 2d pose predictions from StackedHourglass")
FLAGS["action"] = "All" # "The action to train on. 'All' means all the actions")

# Architecture
FLAGS["linear_size"] = 1024 # "Size of each model layer.")
FLAGS["num_layers"] = 2 # "Number of layers in the model.")
FLAGS["residual"] = True # "Whether to add a residual connection every 2 layers")

# Evaluation
FLAGS["procrustes"] = False # "Apply procrustes analysis at test time")
FLAGS["evaluateActionWise"] = True # "The dataset to use either h36m or heva")

# Directories
FLAGS["cameras_path"] = "../data/h36m/cameras.h5" # "Directory to load camera parameters")
FLAGS["data_dir"] = "../data/h36m/" # "Data directory")
FLAGS["train_dir"] = "../experiments/test_git/" # "Training directory.")
FLAGS["load_dir"] = "../Models/mdm_5_prior/" # "Specify the directory to load trained model")

# Train or load
FLAGS["sample"] = False # "Set to True for sampling.")
FLAGS["test"] = False # "Set to True for sampling.")
FLAGS["use_cpu"] = False # "Whether to use the CPU")
FLAGS["load"] = 0 # "Try to load a previous checkpoint.")
FLAGS["miss_num"] = 1 # "Specify how many missing joints.")

# Misc
FLAGS["use_fp16"] = False # "Train using fp16 instead of fp32.")
FLAGS["n_joints"] = 16

summaries_dir = ""
logdir = os.path.join(FLAGS["train_dir"],"log")
os.system('mkdir -p {}'.format(summaries_dir))

eval_every_n_epochs = 10

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, help="Experiment and checkpoint path")
    parser.add_argument("--test", type=bool)
    parser.add_argument("--load", type=int)
    parser.add_argument("--load_dir", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--eval_n_epoch", type=int, default=5)
    parser.add_argument("--qual", type=str, default=None)
    parser.add_argument("--metrics", type=str, default=None)
    
    return parser.parse_args()


def create_model(session, batch_size):

    model = LinearModel(
      FLAGS["linear_size"],
      FLAGS["num_layers"],
      FLAGS["residual"],
      FLAGS["batch_norm"],
      FLAGS["max_norm"],
      batch_size,
      FLAGS["learning_rate"],
      summaries_dir,
      FLAGS["predict_14"],
      dtype=tf.float16 if FLAGS["use_fp16"] else tf.float32)

    if FLAGS['load'] <= 0:
        session.run(tf.global_variables_initializer())
        return model

    ckpt = tf.train.get_checkpoint_state(FLAGS['load_dir'], latest_filename='checkpoint')

    if ckpt and ckpt.model_checkpoint_path:
        if FLAGS['load'] > 0:
            if os.path.isfile(os.path.join(FLAGS['load_dir'], "checkpoint-{0}.index".format(FLAGS['load']))):
                ckpt_name = os.path.join(FLAGS['load_dir'], "checkpoint-{0}".format(FLAGS["load"]))
            else:
                raise ValueError("Checkpoint {0} does not exist in directory {1}".format(FLAGS['load'], FLAGS['load_dir']))
        else:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

        print("Loading checkpoint {0}".format(ckpt_name))
        model.saver.restore(session, ckpt_name)

        return model
    else:
        raise ValueError("Problem loading checkpoint")


def train(dataset):

    train_2d = dataset.get_2d_train()
    train_3d = dataset.get_3d_train()

    device_count = { "GPU": 1 }

    with tf.Session(config = tf.ConfigProto(
            device_count = device_count,
            allow_soft_placement = True
        )) as sess:

        model = create_model(sess, FLAGS["batch_size"])

        model.train_writer.add_graph( sess.graph )

        step_time, val_loss, last_step_loss = 0.0, 0.0, 0.0
        start_time = time.time()

        current_epoch = 0
        current_step = 0
        log_every_n_batches = 100
        best_val_so_far = np.inf

        n_epochs = FLAGS['epochs']

        once = False

        for epoch in range(n_epochs):
            loss = AverageMeter()

            start_time = time.time()
            current_epoch += 1

            print(f"Epoch {epoch+1}/{n_epochs}:")

            enc_in, dec_out = model.get_all_batches(train_2d, train_3d, camera_frame=FLAGS["camera_frame"], training=True)

            n_batches = len( enc_in )

            bar = Bar('Train', max=n_batches)
            for i_batch in range(n_batches):

                in_, out_ = enc_in[i_batch], dec_out[i_batch]

                step_loss, loss_summary, lr_summary, comp = model.step(sess, in_, out_, 0.5, True)

                loss.update(step_loss)
                step_time = time.time() - start_time
                start_time = time.time()

                if i_batch % log_every_n_batches == 0:
                    last_step_loss = step_loss
                    model.train_writer.add_summary(loss_summary, current_step)
                    model.train_writer.add_summary(lr_summary, current_step)

                bar.suffix = f'Batch: {i_batch}/{n_batches} | Loss: {loss.avg:.2f} | Last Step Loss: {last_step_loss:.2f} | Step Time: {step_time*1000:.2f}'
                bar.next()

                current_step += 1

            bar.finish()

            if (epoch+1) % eval_every_n_epochs == 0:
                err = evaluate(sess, model, dataset)
                print(f"Evaluation reuslt> MPJPE: {err} mm")
                save_time = time.time()
                if err < best_val_so_far:
                    best_val_so_far = err
                    model.saver.save(sess, os.path.join(FLAGS['train_dir'], 'checkpoint'), global_step=current_step)
                    save_time -= time.time()
                    print(f'Model saving completed in {save_time} ms')


def test(dataset):

    device_count = { "GPU": 1 }

    with tf.Session(config = tf.ConfigProto(
            device_count = device_count,
            allow_soft_placement = True
        )) as sess:

        model = create_model(sess, FLAGS["batch_size"])

        err = evaluate(sess, model, dataset)


def evaluate(sess, model, dataset):
    test_2d = dataset.get_2d_valid()
    test_3d = dataset.get_3d_valid()

    enc_in, dec_out = model.get_all_batches(test_2d, test_3d, camera_frame=False, training=False)

    n_batches = len( enc_in )

    mpjpe = AverageMeter()

    all_dists = []

    bar = Bar("Eval", max=n_batches)

    for i_batch in range(n_batches):

        # d_temp = np.load("data/temp.npz", allow_pickle=True)

        in_, out_ = enc_in[i_batch], dec_out[i_batch]

        # in_, out_ = d_temp['enc_in'], d_temp['dec_out']

        step_loss, loss_summary, pred_all = model.step(sess, in_, out_, dropout_keep_prob=1.0, isTraining=False)

        pred_all_re = np.reshape(pred_all, [-1, model.HUMAN_3D_SIZE+2, model.num_models])
        pred_all_re = pred_all_re[:, :model.HUMAN_3D_SIZE, :]
        # print(pred_all_re.shape)
        pred_all_re = pred_all_re.reshape((-1, model.HUMAN_3D_SIZE//3, 3, model.num_models))

        pose_3d = np.zeros(pred_all_re.shape)
        # print(pose_3d.shape)

        for k in range(pose_3d.shape[-1]):
            pose_3d[:, :, :, k] = unnormalize_data(pred_all_re[:, :, :, k], dataset.mean_3d, dataset.std_3d, skip_root=True)

        out_re = out_.reshape((-1, model.HUMAN_3D_SIZE//3, 3))

        un_out = unnormalize_data(out_re, dataset.mean_3d, dataset.std_3d, skip_root=True)

        sqerr = (pose_3d - np.expand_dims(un_out, axis=3))**2
        dists = np.zeros((sqerr.shape[0], model.HUMAN_3D_SIZE//3, sqerr.shape[3]))

        for m in range(dists.shape[-1]):
            dist_idx = 0
            for k in range(1, model.HUMAN_3D_SIZE//3):

                dists[:, dist_idx, m] = np.sqrt(np.sum(sqerr[:, k, :, m], axis=1))

                dist_idx += 1

        temp_mpjpe = np.mean(np.min(np.sum(dists, axis=1), axis=1))/(17)

        mpjpe.update(temp_mpjpe)

        all_dists.append(dists)

        bar.suffix = f'Batch: {i_batch}/{n_batches} | MPJPE: {mpjpe.avg:.2f}'
        bar.next()

    all_dists = np.vstack(all_dists)
    avg_minerr = np.mean(np.min(np.sum(all_dists, axis=1), axis=1))/(17)

    bar.finish()

    return avg_minerr


def qualitative_test(dataset, dirname):

    save_dir = os.path.join("out", dirname)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    test_2d = dataset.get_2d_valid()
    test_3d = dataset.get_3d_valid()

    device_count = { "GPU": 1 }

    with tf.Session(config = tf.ConfigProto(
            device_count = device_count,
            allow_soft_placement = True
        )) as sess:

        model = create_model(sess, FLAGS["batch_size"])

        enc_in, dec_out = model.get_all_batches(test_2d, test_3d, camera_frame=False, training=False)

        n_batches = len(enc_in)

        for i_batch in tqdm(range(n_batches)):
            
            in_, out_ = enc_in[i_batch], dec_out[i_batch]

            _, _, pred_all = model.step(sess, in_, out_, dropout_keep_prob=1.0, isTraining=False)

            pred_all_re = np.reshape(pred_all, [-1, model.HUMAN_3D_SIZE+2, model.num_models])
            pred_all_re = pred_all_re[:, :model.HUMAN_3D_SIZE, :]
            pred_all_re = pred_all_re.reshape((-1, model.HUMAN_3D_SIZE//3, 3, model.num_models))

            pose_3d = np.zeros(pred_all_re.shape)

            for k in range(pose_3d.shape[-1]):
                pose_3d[:, :, :, k] = unnormalize_data(pred_all_re[:, :, :, k], dataset.mean_3d, dataset.std_3d, skip_root=True)
                plot_3d(pose_3d[0, :, :, k], os.path.join(save_dir, str(i_batch)+"_"+str(k)+".png"))
            
            out_re = np.reshape(out_, (-1, model.HUMAN_3D_SIZE//3, 3))

            out_re = unnormalize_data(out_re, dataset.mean_3d, dataset.std_3d, skip_root=True)

            plot_3d(out_re[0, :, :], os.path.join(save_dir, str(i_batch)+"_gt.png"))


def check_flag_consistency(args):

    if not os.path.exists("metrics"):
        os.mkdir("metrics")

    if args.train_dir:
        FLAGS['train_dir'] = os.path.join("experiments", args.train_dir)

        if not os.path.exists(os.path.join("experiments", args.train_dir)):
            os.mkdir(os.path.join("experiments", args.train_dir))

        summaries_dir = os.path.join(FLAGS["train_dir"], "summaries")

        if not os.path.exists(summaries_dir):
            os.mkdir(summaries_dir)
    
    if args.test:
        FLAGS['test'] = args.test

    if args.load:
        FLAGS['load'] = args.load

    if args.load_dir:
        FLAGS['load_dir'] = os.path.join("experiments", args.load_dir)
        FLAGS['train_dir'] = os.path.join("experiments", args.load_dir)
        summaries_dir = os.path.join(FLAGS["load_dir"], "summaries")


if __name__ == "__main__":

    args = parser()
    check_flag_consistency(args)
    eval_every_n_epochs = args.eval_n_epoch
    
    if args.dataset == '3dpw':
        dataset = ThreeDPWDataset('data/3dpw_wo_invalid.npz', args.metrics)
    elif args.dataset == 'gpa':
        dataset = GPADataset('data/gpa_xyz_projected_wc_v3.npz', args.metrics)
    elif args.dataset == 'h36m':
        dataset = H36MDataset('data/h36m', args.metrics)
    elif args.dataset == 'surreal':
        dataset = SurrealDataset("data/surreal_train_compiled.npz", "data/surreal_val_compiled.npz")
    elif args.dataset == '3dpw_aug':
        dataset = ThreeDPWDataset("data/3dpw_augmented_wo_invalid.npz", args.metrics)
    else:
        raise ValueError("Dataset not supported. Only supports: h36m, gpa, and 3dpw")
    
    if args.qual:
        qualitative_test(dataset, args.qual)
    elif FLAGS['test']:
        test(dataset)
    else:
        train(dataset)