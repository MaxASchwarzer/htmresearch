import torch
from torch.nn.functional import conv2d, relu_, unfold, fold
localConv=torch.nn.backends.thnn.backend.SpatialConvolutionLocal
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle as pkl
import seaborn as sns
from sklearn.linear_model import LinearRegression
from Pytorch2DCAN import GCN2D
import argparse

from utils import *

def createMovie(data, name, interval=10):
    def update_line(num, data, line):
        line.set_data(data[num])
        return line,

    fig, ax   = plt.subplots(1,1)

    a0  = ax.matshow(data[0], animated=True,)
    ani = animation.FuncAnimation(fig, update_line, data.shape[0] - 1,
                                  fargs=(data, a0),
                                  interval=interval, blit=False)

    mywriter = animation.FFMpegWriter()
    ani.save(name,writer=mywriter)

    return(ani)

def train(args):
    if args.load is None:
        GCN = GCN2D(numX=args.numX,
                    numY=args.numY,
                    inhibitionWindowX=args.inhibitionWindowX,
                    inhibitionWindowY=args.inhibitionWindowY,
                    inhibitionRadius=args.inhibitionRadius,
                    inhibitionStrength=args.inhibitionStrength,
                    excitationWindow=args.excitationWindow,
                    wideningFactor=args.wideningFactor,
                    excitationCenterBlock=args.excitationCenterBlock,
                    globalTonic=args.globalTonic,
                    stdpWindow=args.stdpWindow,
                    dt=args.dt,
                    placeMax=args.placeMax,
                    placeMean=args.placeMean,
                    boostEffect=args.boostEffect,
                    boostDecay=args.boostDecay,
                    numPlaces=args.numPlaces,
                    circularPlaces=args.circularPlaces,
                    learningRate=args.learningRate,
                    initialWeightFactor=args.initialWeightFactor,
                    initialExcitatoryFactor=args.initialExcitatoryFactor,
                    boostGradientX=args.boostGradientX,
                    decayGradientX=args.decayGradientX,
                    tonicGradientX=args.tonicGradientX,
                    inhibitionGradientX=args.inhibitionGradientX,
                    inhibitoryWeightDecay=args.inhibitoryWeightDecay,
                    decayConstant=args.decayConstant,
                    negativeLearnFactorP=args.negativeLearnFactorP,
                    negativeLearnFactorE=args.negativeLearnFactorE,
                    negativeLearnFactorI=args.negativeLearnFactorI,
                    learningRateE=args.learningRate,
                    envelopeWidth=args.envelopeWidth,
                    envelopeFactor=args.envelopeFactor,
                    sigmaLoc=args.sigmaLoc,
                    hardwireE=args.hardwireE,
                    hardwireEStrength=args.hardwireEStrength,
                    hardwireERange=args.hardwireERange,
                    hardwireEOffset=args.hardwireEOffset,
                    excitationGradientX=args.excitationGradientX,
                    gradientType=args.gradientType,
                    envSize=args.envSize,
                    weightNoise=args.weightNoise)
    else:
        GCN = pkl.load(args.load + ".pkl")

    GCN.randomLesions(args.lesions,
                      args.lesionOuterCutoff,
                      args.lesionInnerCutoff)


    for i in range(args.iters):
        print("Iteration {}".format(i))
        GCN.staticLearning(args.length,
                            logFreq=100000,
                            startFrom = args.warmup)

    results = GCN.simulate(25, logFreq=1, startFrom = 0, vel = (0, 0))

    with open(args.save + "_results.pkl", "wb") as f:
        pkl.dump(results, f)

    with open(args.save + ".pkl", "wb") as f:
        pkl.dump(GCN, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', type=int, default=200,
                        help='Training time (s) per iteration')
    parser.add_argument('--warmup', type=int, default=50,
                        help='Warmup time (s) per iteration')
    parser.add_argument('--iters', type=int, default=25,
                        help='Number of iterations')
    parser.add_argument('--save', type=str,  default='gcn2d',
                        help='path to save the final model')
    parser.add_argument('--load', type=str,  default=None,
                        help='path to load a model from to resume training. ' +
                        'Keep blank for none.')

    parser.add_argument('--numX',type=int, default=256,)
    parser.add_argument('--numY',type=int, default=64,)
    parser.add_argument('--inhibitionWindowX',type=int, default=100,)
    parser.add_argument('--inhibitionWindowY',type=int, default=32,)
    parser.add_argument('--inhibitionRadius',type=float, default=.75,)
    parser.add_argument('--inhibitionStrength',type=float, default=450.,)
    parser.add_argument('--excitationWindow',type=int, default=3,)
    parser.add_argument('--wideningFactor', type=float, default=2,)
    parser.add_argument('--excitationCenterBlock', type=float, default=-1,)
    parser.add_argument('--globalTonic', type=float, default=.1,)
    parser.add_argument('--stdpWindow', type=int, default=2,)
    parser.add_argument('--dt', type=float, default=0.01,)
    parser.add_argument('--placeMax', type=float, default=1.,)
    parser.add_argument('--placeMean', type=float, default=.25,)
    parser.add_argument('--boostEffect', type=float, default=10.,)
    parser.add_argument('--boostDecay', type=float, default=.1,)
    parser.add_argument('--numPlaces', type=int, default=1000,)
    parser.add_argument('--circularPlaces', action='store_true')
    parser.add_argument('--learningRate', type=float, default=.01,)
    parser.add_argument('--initialWeightFactor', type=float, default=.02,)
    parser.add_argument('--initialExcitatoryFactor', type=float, default=0.05,)
    parser.add_argument('--boostGradientX', type=float, default=1.,)
    parser.add_argument('--decayGradientX', type=float, default=30.,)
    parser.add_argument('--tonicGradientX', type=float, default=1.,)
    parser.add_argument('--inhibitionGradientX', type=float, default=1.,)
    parser.add_argument('--inhibitoryWeightDecay', type=float, default=5000,)
    parser.add_argument('--decayConstant', type=float, default=0.03,)
    parser.add_argument('--negativeLearnFactorP', type=float, default=.9,)
    parser.add_argument('--negativeLearnFactorE', type=float, default=1.2,)
    parser.add_argument('--negativeLearnFactorI', type=float, default=5.,)
    parser.add_argument('--learningRateE', type=float, default=0,)
    parser.add_argument('--envelopeWidth', type=float, default=32,)
    parser.add_argument('--envelopeFactor', type=float, default=3.5,)
    parser.add_argument('--sigmaLoc', type=float, default=(.15)**2,)
    parser.add_argument('--hardwireE', action='store_true')
    parser.add_argument('--hardwireEStrength', type=float, default=1.,)
    parser.add_argument('--hardwireERange', type=float, default=.05,)
    parser.add_argument('--hardwireEOffset', type=float, default=2.,)
    parser.add_argument('--excitationGradientX', type=float, default=10.,)
    parser.add_argument('--gradientType', type=float, default=.7,)
    parser.add_argument('--envSize', type=float, default=.5,)
    parser.add_argument('--weightNoise', type=float, default=2.)
    parser.add_argument('--lesions', type=int, default=0)
    parser.add_argument('--lesionInnerCutoff', type=float, default=5.)
    parser.add_argument('--lesionOuterCutoff', type=float, default=15.)
    parser.add_argument('--videoFrameLength', type=float, default=10.)


    args = parser.parse_args()
    print("Training with {}".format(args))
    train(args)
