from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from chainer import serializers
from chainer import Chain
import chainer.links as L
import tensorflow as tf
from PIL import Image
import numpy as np
import cupy
import pickle
import os

from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
from keras_vggface import utils

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)


class ModelLinear(Chain):
    def __init__(self, n_in, n_out):
        super(ModelLinear, self).__init__()
        with self.init_scope():
            self.fc1 = L.Linear(n_in, n_out)

    def __call__(self, x):
        return self.fc1(x)


def load_network(dev):
    p = "karras2018iclr-celebahq-1024x1024.pkl"
    tf.InteractiveSession()
    with tf.device('/gpu:%d' % dev):
        _, _, Gs = pickle.load(open(p, "rb"))
    return Gs


def face_from_latent(model, latents, my_path, save_image=True):
    dummy_label = np.zeros([1] + model.input_shapes[1][1:])
    for i in range(latents.shape[0]):
        latent = np.expand_dims(latents[i], 0)
        face = model.run(latent, dummy_label)
        face = np.clip(np.rint((face + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)
        face = face.transpose((0, 2, 3, 1))

        if save_image:
            if not os.path.exists(my_path):
                os.mkdir(my_path)
            save_path = os.path.join(my_path, '%d.png' % i)
            Image.fromarray(face[0], 'RGB').save(save_path)
        else:
            Image.fromarray(face[0], 'RGB').show()


def get_features(img):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=2)
    feats = vgg_features.predict(x)
    return feats[0][0][0]


if __name__ == '__main__':

    np.random.seed(1412)
    dev = 1
    with tf.device('/gpu:%d' % dev):
        model = load_network(dev=dev)

    # data
    with open('data_sub1_4096.dat', 'rb') as fp:
        _, _, X_test, T_test = pickle.load(fp)

    # predict latents
    with cupy.cuda.Device(dev):
        model_linear = ModelLinear(n_in=4096, n_out=512).to_gpu(dev)
        serializers.load_npz('l0_s1_5000_final.model', model_linear)
        X_test = cupy.array(X_test, dtype=cupy.float32)
        T_test = cupy.array(T_test, dtype=cupy.float32)
        Y_test = model_linear(X_test).array

    # generate stimuli and reconstructions
    face_from_latent(model, cupy.asnumpy(T_test), 'stimuli', save_image=True)
    face_from_latent(model, cupy.asnumpy(Y_test), 'reconstructions', save_image=True)

    # stimuli vs. reconstructions
    trials = len(T_test)
    metrics = {"lsim": np.zeros((trials, )),
               "fsim": np.zeros((trials, )),
               "ssim": np.zeros((trials, )),
               "gender": np.zeros((trials, )),
               "age": np.zeros((trials, )),
               "eyeglasses": np.zeros((trials, )),
               "pose": np.zeros((trials, )),
               "smile": np.zeros((trials, ))}

    test_feats = np.zeros((36, 2048))
    pred_feats = np.zeros((36, 2048))
    test_scores = np.zeros((5, 36))
    pred_scores = np.zeros((5, 36))
    vgg_features = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))

    gender = np.load('bounds/pggan_celebahq_gender_boundary.npy')
    age = np.load('bounds/pggan_celebahq_age_boundary.npy')
    eyeglasses = np.load('bounds/pggan_celebahq_eyeglasses_boundary.npy')
    pose = np.load('bounds/pggan_celebahq_pose_boundary.npy')
    smile = np.load('bounds/pggan_celebahq_smile_boundary.npy')
    boundaries = [gender, age, eyeglasses, pose, smile]

    Y_test = cupy.asnumpy(Y_test)
    T_test = cupy.asnumpy(T_test)
    for trial in range(trials):
        stim_ssim = np.array(Image.open("stimuli/%i.png" % trial))
        recon_ssim = np.array(Image.open("reconstructions/%i.png" % trial))
        stim_fsim = image.load_img("stimuli/%i.png" % trial, target_size=(224, 224))
        recon_fsim = image.load_img("reconstructions/%i.png" % trial, target_size=(224, 224))

        test_feats[trial] = get_features(stim_fsim)
        pred_feats[trial] = get_features(recon_fsim)
        metrics['lsim'][trial] = 1. / (1 + mean_squared_error(Y_test[trial], T_test[trial]))
        metrics['fsim'][trial] = 1. / (1 + mean_squared_error(test_feats[trial], pred_feats[trial]))
        metrics['ssim'][trial] = ssim(stim_ssim, recon_ssim, multichannel=True)

        for i, boundary in enumerate(boundaries):
            test_scores[i, trial] = T_test[trial].reshape(1, -1).dot(boundary.T)[0][0]
            pred_scores[i, trial] = Y_test[trial].reshape(1, -1).dot(boundary.T)[0][0]

    # print metrics
    print("latent similarity: %.4f" % metrics['lsim'].mean())
    print("Feature similarity: %.4f" % metrics['fsim'].mean())
    print("Structural similarity: %.4f" % metrics['ssim'].mean())

    names = ["Gender", "Age", "Eyeglasses", "Pose", "Smile"]
    for i in range(5):
        corr, pval = pearsonr(test_scores[i], pred_scores[i])
        print("%s Corr. coef.: %.4f" % (names[i], corr))

