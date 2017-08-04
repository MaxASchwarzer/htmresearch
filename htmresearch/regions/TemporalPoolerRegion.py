# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import numpy

from nupic.bindings.math import GetNTAReal
from nupic.support import getArgumentDescriptions
from nupic.bindings.regions.PyRegion import PyRegion

from htmresearch.algorithms.union_temporal_pooler import UnionTemporalPooler
from htmresearch.algorithms.simple_union_pooler import SimpleUnionPooler
from nupic.bindings.algorithms import SpatialPooler
from htmresearch.support.union_temporal_pooler_monitor_mixin import (
  UnionTemporalPoolerMonitorMixin)

class MonitoredUnionTemporalPooler(UnionTemporalPoolerMonitorMixin,
  UnionTemporalPooler): pass

uintDType = "uint32"


class TemporalPoolerRegion(PyRegion):
  """
  The TemporalPoolerRegion implements an L2 layer within a single cortical column / cortical
  module.

  The layer supports feed forward (proximal) and lateral inputs.
  """

  @classmethod
  def getSpec(cls):
    """
    Return the Spec for TemporalPoolerRegion.

    The parameters collection is constructed based on the parameters specified
    by the various components (tmSpec and otherSpec)
    """
    spec = dict(
      description=TemporalPoolerRegion.__doc__,
      singleNodeOnly=True,
      inputs=dict(
        activeCells=dict(
          description="Active cells",
          dataType="UInt32",
          count=0,
          required=True,
          regionLevel=False,
          isDefaultInput=True,
          requireSplitterMap=False),

        predictedActiveCells=dict(
          description="Predicted Active Cells",
          dataType="UInt32",
          count=0,
          required=True,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False),

        lateralInput=dict(
          description="Lateral binary input into this column, presumably from"
                      " other neighboring columns.",
          dataType="UInt32",
          count=0,
          required=False,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False),

        apicalInput=dict(
          description="apical feedback input",
          dataType="UInt32",
          count=0,
          required=False,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False),

        predictedCells=dict(
          description="Predicted Cells",
          dataType="UInt32",
          count=0,
          required=False,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False),

        winnerCells=dict(
          description="Winner Cells",
          dataType="UInt32",
          count=0,
          required=False,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False),

        resetIn=dict(
          description="""A boolean flag that indicates whether
                         or not the input vector received in this compute cycle
                         represents the start of a new temporal sequence.""",
          dataType='Real32',
          count=1,
          required=False,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False),

        sequenceIdIn=dict(
          description="Sequence ID",
          dataType='UInt64',
          count=1,
          required=False,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False),

      ),
      outputs=dict(
        mostActiveCells=dict(
          description="Most active cells in the pooler SDR having non-zero activation",
          dataType="UInt32",
          count=0,
          regionLevel=True,
          isDefaultOutput=True),
        currentlyActiveCells=dict(
          description="Cells in the pooler SDR which became active in the most recent time step",
          dataType="UInt32",
          count=0,
          regionLevel=True,
          isDefaultOutput=False),
        predictedActiveCells=dict(
          description="Cells in the pooler SDR which became active in the most "
                      "recent time step with at least one active lateral "
                      "segment.  Only includes cells which were not previously "
                      "active",
          dataType="UInt32",
          count=0,
          regionLevel=True,
          isDefaultOutput=False),
      ),
      parameters=dict(
        learningMode=dict(
          description="Whether the node is learning (default True).",
          accessMode="ReadWrite",
          dataType="Bool",
          count=1,
          defaultValue="true"),
        inputWidth=dict(
          description='Number of inputs to the layer.',
          accessMode='Read',
          dataType='UInt32',
          count=1,
          constraints=''),
        maxUnionActivity=dict(
          description="The number of active cells invoked per object",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints=""),
        columnCount=dict(
          description="Total number of columns (coincidences).",
          accessMode="ReadWrite",
          dataType="UInt32",
          count=1,
          constraints=""),
        apicalInputWidth=dict(
          description="Size of apical input to the UP.",
          accessMode="ReadWrite",
          dataType="UInt32",
          count=1,
          constraints=""),
        lateralInputWidths=dict(
          description="Size of lateral input to the UP.",
          accessMode="ReadWrite",
          dataType="UInt32",
          count=0,
          constraints=""),
        useInternalLateralConnections=dict(
          description="1 if the pooler is forming internal lateral connections",
          accessMode="ReadWrite",
          dataType="UInt32",
          count=1,
          constraints="bool"),
        numActive=dict(
          description="Number of active cells per time step",
          accessMode="ReadWrite",
          dataType="UInt32",
          count=1,
          constraints=""),
        historyLength=dict(
          description="The union window length",
          accessMode="ReadWrite",
          dataType="UInt32",
          count=1,
          constraints=""),
        minHistory=dict(
          description="don't perform union (output all zeros) until buffer"
                      " length >= minHistory",
          accessMode="ReadWrite",
          dataType="UInt32",
          count=1,
          constraints=""),
        poolerType=dict(
          description="Type of pooler to use: union",
          accessMode="ReadWrite",
          dataType="Byte",
          count=0,
          constraints="enum: union"),

        #
        # Proximal
        #
        synPermActiveInc=dict(
          description="Amount by which permanences of proximal synapses are "
                      "incremented during learning.",
          accessMode="Read",
          dataType="Real32",
          count=1),
        synPermInactiveDec=dict(
          description="Amount by which permanences of proximal synapses are "
                      "decremented during learning.",
          accessMode="Read",
          dataType="Real32",
          count=1),
        synPermPredActiveInc=dict(
          description="Amount by which permanences of proximal synapses are "
                      "incremented when learning predicted active input.",
          accessMode="Read",
          dataType="Real32",
          count=1),
        synPermPreviousPredActiveInc=dict(
          description="Amount by which permanences of proximal synapses are "
                      "incremented to when learning previously active input.",
          accessMode="Read",
          dataType="Real32",
          count=1),
        potentialPct=dict(
          description="Fraction of input bits that each cellcan connect to",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints=""),
        stimulusThreshold=dict(
          description="If the number of synapses active on a proximal segment "
                      "is at least this threshold, it can be considered as a "
                      "candidate active cell",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        synPermConnected=dict(
          description="If the permanence value for a synapse is greater "
                      "than this value, it is said to be connected.",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints=""),
        activeOverlapWeight=dict(
          description="Value of each active input bit",
          accessMode="Read",
          dataType="Real32",
          count=1),
        predictedActiveOverlapWeight=dict(
          description="Value of each predicted active input bit",
          accessMode="Read",
          dataType="Real32",
          count=1),

        #
        # Distal
        #
        synPermDistalInc=dict(
          description="Amount by which permanences of synapses are "
                      "incremented during learning.",
          accessMode="Read",
          dataType="Real32",
          count=1),
        synPermDistalDec=dict(
          description="Amount by which permanences of synapses are "
                      "decremented during learning.",
          accessMode="Read",
          dataType="Real32",
          count=1),
        initialDistalPermanence=dict(
          description="Initial permanence of a new synapse.",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints=""),
        sampleSizeDistal=dict(
          description="The desired number of active synapses for an active "
                      "segment.",
          accessMode="Read",
          dataType="Int32",
          count=1),
        activationThresholdDistal=dict(
          description="If the number of synapses active on a distal segment is "
                      "at least this threshold, the segment is considered "
                      "active",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        connectedPermanenceDistal=dict(
          description="If the permanence value for a synapse is greater "
                      "than this value, it is said to be connected.",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints=""),
        distalSegmentBoost=dict(
          description="Controls how powerful a boost each cell gets for having "
                      "an active segment.  Scales multiplicatively.",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints=""),


        # Apical
        synPermApicalInc=dict(
          description="Amount by which permanences of synapses are "
                      "incremented during learning.",
          accessMode="Read",
          dataType="Real32",
          count=1),
        synPermApicalDec=dict(
          description="Amount by which permanences of synapses are "
                      "decremented during learning.",
          accessMode="Read",
          dataType="Real32",
          count=1),
        initialApicalPermanence=dict(
          description="Initial permanence of a new synapse.",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints=""),
        sampleSizeApical=dict(
          description="The desired number of active synapses for an active "
                      "segment.",
          accessMode="Read",
          dataType="Int32",
          count=1),
        activationThresholdApical=dict(
          description="If the number of synapses active on a distal segment is "
                      "at least this threshold, the segment is considered "
                      "active",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        connectedPermanenceApical=dict(
          description="If the permanence value for a synapse is greater "
                      "than this value, it is said to be connected.",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints=""),
        apicalSegmentBoost=dict(
          description="Controls how powerful a boost each cell gets for having "
                      "an active segment.  Scales multiplicatively.",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints=""),



        inhibitionFactor=dict(
          description="Controls how strongly cells inhibit each other.",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints=""),

        # Misc
        seed=dict(
          description="Seed for the random number generator.",
          accessMode="Read",
          dataType="UInt32",
          count=1),
        wrapAround=dict(
          description="Seed for the random number generator.",
          accessMode="Read",
          dataType="UInt32",
          count=1),
        boostStrength=dict(
          description="Controls how strongly boosting is applied",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints=""),
        decayFunctionType=dict(
          description="Type of decay function to use",
          accessMode="ReadWrite",
          dataType="Byte",
          count=0,
          constraints=""),
        exciteFunctionType=dict(
          description="Type of excite function to use",
          accessMode="ReadWrite",
          dataType="Byte",
          count=0,
          constraints=""),
        decayTimeConst=dict(
          description="Controls the timescale on which decay is applied",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        dutyCyclePeriod=dict(
          description="The duty cycle used for boosting",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        minPctOverlapDutyCycle=dict(
          description="Minimum overlap for duty cycle in boosting",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints=""),
        globalInhibition=dict(
          description="Whether or not the SP should use global inhibition.",
          accessMode="ReadWrite",
          dataType="UInt32",
          count=1,
          constraints="bool"),
      ),
      commands=dict()
    )
    return spec


  def __init__(self,
               # union_temporal_pooler.py parameters
               activeOverlapWeight=1.0,
               predictedActiveOverlapWeight=0.0,
               numActive = 40,
               columnCount = 2048,

               # Distal
               distalSegmentBoost = 1.2,
               lateralInputWidths = [],
               useInternalLateralConnections = False,
               synPermDistalInc=0.1,
               synPermDistalDec=0.001,
               initialDistalPermanence=0.6,
               sampleSizeDistal=20,
               activationThresholdDistal=13,
               connectedPermanenceDistal=0.50,

               # Apical
               apicalSegmentBoost=2.5,
               apicalInputWidth=None,
               synPermApicalInc=0.1,
               synPermApicalDec=0.001,
               initialApicalPermanence=0.6,
               sampleSizeApical=20,
               activationThresholdApical=13,
               connectedPermanenceApical=0.50,

               exciteFunctionType='Fixed',
               decayFunctionType='NoDecay',
               maxUnionActivity=0.20,
               decayTimeConst=20.0,
               synPermPredActiveInc=0.0,
               synPermPreviousPredActiveInc=0.0,
               historyLength=0,
               minHistory=0,
               inhibitionFactor = 1.,

               # Spatial Pooler
               inputWidth = 2048*8,
               potentialRadius=16,
               potentialPct=0.5,
               globalInhibition=True,
               stimulusThreshold=0,
               synPermInactiveDec=0.008,
               synPermActiveInc=0.05,
               synPermConnected=0.10,
               minPctOverlapDutyCycle=0.001,
               dutyCyclePeriod=1000,
               boostStrength=0.0,
               spVerbosity=0,
               wrapAround=True,
               #spVersion = "c++",
               seed = 42,
               learningMode = True,
               **kwargs):

    self.activeOverlapWeight = activeOverlapWeight
    self.predictedActiveOverlapWeight = predictedActiveOverlapWeight
    self.numActive  = numActive
    self.columnCount = columnCount

    # Distal
    self.distalSegmentBoost  = distalSegmentBoost
    self.lateralInputWidths  = lateralInputWidths
    self.useInternalLateralConnections  = useInternalLateralConnections
    self.synPermDistalInc = synPermDistalInc
    self.synPermDistalDec = synPermDistalDec
    self.initialDistalPermanence = initialDistalPermanence
    self.sampleSizeDistal = sampleSizeDistal
    self.activationThresholdDistal = activationThresholdDistal
    self.connectedPermanenceDistal = connectedPermanenceDistal

    # Apical
    self.apicalInputWidth  = apicalInputWidth
    self.synPermApicalInc = synPermApicalInc
    self.synPermApicalDec = synPermApicalDec
    self.initialApicalPermanence = initialApicalPermanence
    self.sampleSizeApical = sampleSizeApical
    self.activationThresholdApical = activationThresholdApical
    self.connectedPermanenceApical = connectedPermanenceApical
    self.apicalSegmentBoost  = apicalSegmentBoost

    self.exciteFunctionType = exciteFunctionType
    self.decayFunctionType = decayFunctionType
    self.maxUnionActivity = maxUnionActivity
    self.decayTimeConst = decayTimeConst
    self.synPermPredActiveInc = synPermPredActiveInc
    self.synPermPreviousPredActiveInc = synPermPreviousPredActiveInc
    self.historyLength = historyLength
    self.minHistory = minHistory
    self.inhibitionFactor = inhibitionFactor

    # Spatial Pooler
    self.inputWidth  = inputWidth
    self.potentialRadius = potentialRadius
    self.potentialPct = potentialPct
    self.globalInhibition = globalInhibition
    self.stimulusThreshold = stimulusThreshold
    self.synPermInactiveDec = synPermInactiveDec
    self.synPermActiveInc = synPermActiveInc
    self.synPermConnected = synPermConnected
    self.minPctOverlapDutyCycle = minPctOverlapDutyCycle
    self.dutyCyclePeriod = dutyCyclePeriod
    self.boostStrength = boostStrength
    self.spVerbosity = spVerbosity
    self.wrapAround = wrapAround
    #self.#spVersion  = spVersion
    self.seed  = seed
    self.learningMode = learningMode
    self._pooler = None

    PyRegion.__init__(self, **kwargs)


  def initialize(self):
    """
    Initialize the internal objects.
    """
    if self._pooler is None:
      params = {
        "activeOverlapWeight": self.activeOverlapWeight,
        "predictedActiveOverlapWeight": self.predictedActiveOverlapWeight,
        "numActive": self.numActive,
        "columnCount": self.columnCount,
        "distalSegmentBoost": self.distalSegmentBoost,
        "lateralInputWidths": self.lateralInputWidths,
        "useInternalLateralConnections": self.useInternalLateralConnections,
        "synPermDistalInc": self.synPermDistalInc,
        "synPermDistalDec": self.synPermDistalDec,
        "initialDistalPermanence": self.initialDistalPermanence,
        "sampleSizeDistal": self.sampleSizeDistal,
        "activationThresholdDistal": self.activationThresholdDistal,
        "connectedPermanenceDistal": self.connectedPermanenceDistal,

        "apicalSegmentBoost": self.apicalSegmentBoost,
        "synPermApicalInc": self.synPermApicalInc,
        "synPermApicalDec": self.synPermApicalDec,
        "initialApicalPermanence": self.initialApicalPermanence,
        "sampleSizeApical": self.sampleSizeApical,
        "apicalInputWidth": self.apicalInputWidth,
        "activationThresholdApical": self.activationThresholdApical,
        "connectedPermanenceApical": self.connectedPermanenceApical,

        "exciteFunctionType": self.exciteFunctionType,
        "decayFunctionType": self.decayFunctionType,
        "maxUnionActivity": self.maxUnionActivity,
        "decayTimeConst": self.decayTimeConst,
        "synPermPredActiveInc": self.synPermPredActiveInc,
        "synPermPreviousPredActiveInc": self.synPermPreviousPredActiveInc,
        "historyLength": self.historyLength,
        "minHistory": self.minHistory,
        "inhibitionFactor": self.inhibitionFactor,
        "inputWidth": self.inputWidth,
        "potentialRadius": self.potentialRadius,
        "potentialPct": self.potentialPct,
        "globalInhibition": self.globalInhibition,
        "stimulusThreshold": self.stimulusThreshold,
        "synPermInactiveDec": self.synPermInactiveDec,
        "synPermActiveInc": self.synPermActiveInc,
        "synPermConnected": self.synPermConnected,
        "minPctOverlapDutyCycle": self.minPctOverlapDutyCycle,
        "dutyCyclePeriod": self.dutyCyclePeriod,
        "boostStrength": self.boostStrength,
        "spVerbosity": self.spVerbosity,
        "wrapAround": self.wrapAround,
        #"spVersion": self.#spVersion,
        "seed": self.seed
      }
      self._pooler = UnionTemporalPooler(**params)

  def compute(self, inputs, outputs):
    """
    Run one iteration of TemporalPoolerRegion's compute.

    Note that if the reset signal is True (1) we assume this iteration
    represents the *end* of a sequence. The output will contain the pooled
    representation to this point and any history will then be reset. The output
    at the next compute will start fresh.
    """

    resetSignal = False
    if 'resetIn' in inputs:
      if len(inputs['resetIn']) != 1:
        raise Exception("resetIn has invalid length")

      if inputs['resetIn'][0] != 0:
        resetSignal = True

    outputs["mostActiveCells"][:] = numpy.zeros(
                                      self.columnCount, dtype=uintDType)



    if "predictedCells" in inputs:
      predictedCells = inputs["predictedCells"]
    else:
      predictedCells = None

    if "winnerCells" in inputs:
      winnerCells = inputs["winnerCells"]
    else:
      winnerCells = None

    predictedActiveCells = inputs["predictedActiveCells"] if (
      "predictedActiveCells" in inputs) else numpy.zeros(self._inputWidth,
                                                         dtype=uintDType)


    if "lateralInput" in inputs:
      lateralInputs = []
      current = 0
      for width in self.lateralInputWidths:
        rawInput = inputs["lateralInput"][current:width]
        lateralInputs.append(numpy.asarray(rawInput.nonzero()[0],
                                           dtype="uint32"))
      lateralInputs = tuple(lateralInputs)
    else:
      lateralInputs = ()

    if "apicalInput" in inputs:
      apicalInput = inputs["apicalInput"]
    else:
      apicalInput = None


    mostActiveCellsIndices = self._pooler.compute(inputs["activeCells"],
                                                  predictedActiveCells,
                                                  self.learningMode,
                                                  predictedCells,
                                                  winnerCells,
                                                  lateralInputs,
                                                  apicalInput)

    outputs["mostActiveCells"][mostActiveCellsIndices] = 1

    outputs["currentlyActiveCells"][:] = numpy.zeros(
                                      self.columnCount, dtype=uintDType)
    outputs["currentlyActiveCells"][self._pooler._getActiveCells()] = 1
    outputs["predictedActiveCells"][:] = numpy.zeros(
                                      self.columnCount, dtype=uintDType)
    outputs["predictedActiveCells"][self._pooler._getPredictedActiveCells()] = 1

    if resetSignal:
        self.reset()


  def reset(self):
    """Reset the history of the underlying pooling class."""
    if self._pooler is not None:
      self._pooler.reset()


  def setParameter(self, parameterName, index, parameterValue):
    """
    Set the value of a Spec parameter. Most parameters are handled
    automatically by PyRegion's parameter set mechanism. The ones that need
    special treatment are explicitly handled here.
    """
    if hasattr(self, parameterName):
      setattr(self, parameterName, parameterValue)
    else:
      raise Exception("Unknown parameter: " + parameterName)


  def getOutputElementCount(self, name):
    return self.columnCount


  def getParameterArrayCount(self, name, index):
    p = self.getParameter(name)
    if (not hasattr(p, "__len__")):
      raise Exception("Attempt to access parameter '%s' as an array but it is not an array" % name)
    return len(p)


  def getParameterArray(self, name, index, a):

    p = self.getParameter(name)
    if (not hasattr(p, "__len__")):
      raise Exception("Attempt to access parameter '%s' as an array but it is not an array" % name)

    if len(p) >  0:
      a[:] = p[:]
