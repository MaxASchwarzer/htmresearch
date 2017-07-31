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

import random
import copy
import numpy
from nupic.bindings.algorithms import SpatialPooler
# Uncomment below line to use python SP
#from nupic.algorithms.spatial_pooler import SpatialPooler
from nupic.bindings.math import GetNTAReal, SparseMatrix, Random
from htmresearch.frameworks.union_temporal_pooling.activation.excite_functions.excite_functions_all import (
  LogisticExciteFunction, FixedExciteFunction)

from htmresearch.frameworks.union_temporal_pooling.activation.decay_functions.decay_functions_all import (
  ExponentialDecayFunction, NoDecayFunction)


REAL_DTYPE = GetNTAReal()
UINT_DTYPE = "uint32"
_TIE_BREAKER_FACTOR = 0.000001



class UnionTemporalPooler(object):
  """
  Experimental Union Temporal Pooler Python implementation. The Union Temporal
  Pooler builds a "union SDR" of the most recent sets of active columns. It is
  driven by active-cell input and, more strongly, by predictive-active cell
  input. The latter is more likely to produce active columns. Such winning
  columns will also tend to persist longer in the union SDR.
  """


  def __init__(self,
               # union_temporal_pooler.py parameters
               activeOverlapWeight=1.0,
               predictedActiveOverlapWeight=0.0,
               numActive = 40,

               # Distal
               segmentBoost = 1.2,
               lateralInputWidths = [],
               useInternalLateralConnections = False,
               synPermDistalInc=0.1,
               synPermDistalDec=0.001,
               initialDistalPermanence=0.6,
               sampleSizeDistal=20,
               activationThresholdDistal=13,
               connectedPermanenceDistal=0.50,

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
               inputDimensions=(32, 32),
               columnDimensions=(64, 64),
               potentialRadius=16,
               potentialPct=0.5,
               globalInhibition=False,
               localAreaDensity=-1.0,
               numActiveColumnsPerInhArea=10.0,
               stimulusThreshold=0,
               synPermInactiveDec=0.008,
               synPermActiveInc=0.05,
               synPermConnected=0.10,
               minPctOverlapDutyCycle=0.001,
               dutyCyclePeriod=1000,
               boostStrength=0.0,
               spVerbosity=0,
               wrapAround=True,
               spVersion = "c++",

               seed = 42):
    """
    Please see spatial_pooler.py in NuPIC for spatial pooler parameters.
    Although the Union Pooler is not explicitly a subclass of the spatial
    pooler, it is strongly dependent on its behavior, and many of the parameters
    are unchanged.

    Class-specific parameters:
    -------------------------------------

    @param  activeOverlapWeight: A multiplicative weight applied to
            the overlap between connected synapses and active-cell input

    @param  predictedActiveOverlapWeight: A multiplicative weight applied to
            the overlap between connected synapses and predicted-active-cell input

    @param  numActive: An int, which indicates roughly how many cells should
            become active at each time step.  Serves the same purpose as
            numActiveColumnsPerInhArea in the spatial pooler.

    @param  segmentBoost: A multiplicative weight applied to the activation of
            each cell, based on how many active distal segments it has.  Should be
            >=1.  Setting this to 1 makes distal segments have no effect.

    @param  lateralInputWidths: A tuple of ints, which indicate the width of all
            lateral input into the pooler (such as from other columns.)

    @param  useInternalLateralConnections: A Boolean, which, if True, causes the
            pooler to form internal lateral connections, which reinforce specific
            activation patterns.

    @param  synPermDistalInc (float)
            Permanence increment for distal synapses

    @param  synPermDistalDec (float)
            Permanence decrement for distal synapses

    @param  sampleSizeDistal (int)
            Number of distal synapses a cell should grow to each lateral
            pattern, or -1 to connect to every active bit

    @param  initialDistalPermanence (float)
            Initial permanence value for distal synapses

    @param  activationThresholdDistal (int)
            Number of active synapses required to activate a distal segment

    @param  connectedPermanenceDistal (float)
            Permanence required for a distal synapse to be connected

    @param  exciteFunction: If fixedPoolingActivationBurst is False,
            this specifies the ExciteFunctionBase used to excite pooling
            activation.

    @param  decayFunction: Specifies the DecayFunctionBase used to decay pooling
            activation.

    @param  maxUnionActivity: Maximum sparsity of the union SDR

    @param  decayTimeConst Time constant for the decay function

    @param  minHistory don't perform union (output all zeros) until buffer
            length >= minHistory
    """

    self.count = 0
    self.representations = {}
    self.overlaps = []
    if "c" in spVersion:
      from nupic.bindings.algorithms import SpatialPooler
    elif "py" in spVersion:
      from nupic.algorithms.spatial_pooler import SpatialPooler
    else:
      print "Unknown spatial pooler version selected."
      raise RuntimeError()
    self.spatialPoolerVersion = SpatialPooler
    self.spatialPooler = SpatialPooler(
      columnDimensions = columnDimensions,
      inputDimensions = inputDimensions,
      numActiveColumnsPerInhArea = numActive,
      seed = seed,
      synPermInactiveDec = synPermInactiveDec,
      synPermActiveInc = synPermActiveInc,
      synPermConnected = synPermActiveInc,
      potentialPct = potentialPct,
      globalInhibition = globalInhibition,
      minPctOverlapDutyCycle = minPctOverlapDutyCycle,
      dutyCyclePeriod = dutyCyclePeriod,
      boostStrength = boostStrength,
      spVerbosity = spVerbosity,
      wrapAround = wrapAround)

    if "c" in spVersion:
      self._updateBookeepingVars = self.spatialPooler.updateBookeepingVars_
      self._updateDutyCycles = self.spatialPooler.updateDutyCycles_
      self._updateBoostFactors = self.spatialPooler.updateBoostFactors_
      self._bumpUpWeakColumns = self.spatialPooler.bumpUpWeakColumns_
      self._isUpdateRound = self.spatialPooler.isUpdateRound_
      self._updateMinDutyCycles = self.spatialPooler.updateMinDutyCycles_
      self._updateInhibitionRadius = self.spatialPooler.updateInhibitionRadius_
      self._updatePermanencesForColumn = self.spatialPooler.updatePermanencesForColumn_

    elif "py" in spVersion:
      self._updateBookeepingVars = self.spatialPooler._updateBookeepingVars
      self._updateDutyCycles = self.spatialPooler._updateDutyCycles
      self._updateBoostFactors = self.spatialPooler._updateBoostFactors
      self._bumpUpWeakColumns = self.spatialPooler._bumpUpWeakColumns
      self._isUpdateRound = self.spatialPooler._isUpdateRound
      self._updateMinDutyCycles = self.spatialPooler._updateMinDutyCycles
      self._updateInhibitionRadius = self.spatialPooler._updateInhibitionRadius

    self.getPermanence = self.spatialPooler.getPermanence
    self.getPotential = self.spatialPooler.getPotential
    self.setOverlapDutyCycles = self.spatialPooler.setOverlapDutyCycles
    self.setActiveDutyCycles = self.spatialPooler.setActiveDutyCycles
    self.setMinOverlapDutyCycles = self.spatialPooler.setActiveDutyCycles
    self.setMinActiveDutyCycles = self.spatialPooler.setActiveDutyCycles
    self.setBoostFactors = self.spatialPooler.setActiveDutyCycles

    self._random = Random()

    self._activeOverlapWeight = activeOverlapWeight
    self._inhibitionFactor = 0.8
    self._predictedActiveOverlapWeight = predictedActiveOverlapWeight
    self._maxUnionActivity = maxUnionActivity
    self._numActive = numActive
    self._exciteFunctionType = exciteFunctionType
    self._decayFunctionType = decayFunctionType
    self._synPermPredActiveInc = synPermPredActiveInc
    self._synPermPreviousPredActiveInc = synPermPreviousPredActiveInc
    self._synPermActiveInc = synPermActiveInc
    self._synPermInactiveDec = synPermInactiveDec
    self._numColumns = numpy.prod(columnDimensions)
    self._numInputs = numpy.prod(inputDimensions)

    self.useInternalLateralConnections = useInternalLateralConnections
    self.segmentBoost = segmentBoost
    self.synPermDistalInc = synPermDistalInc
    self.synPermDistalDec = synPermDistalDec
    self.initialDistalPermanence = initialDistalPermanence
    self.connectedPermanenceDistal = connectedPermanenceDistal
    self.sampleSizeDistal = sampleSizeDistal
    self.activationThresholdDistal = activationThresholdDistal

    self._historyLength = historyLength
    self._minHistory = minHistory

    # initialize excite/decay functions
    if exciteFunctionType == 'Fixed':
      self._exciteFunction = FixedExciteFunction()
    elif exciteFunctionType == 'Logistic':
      self._exciteFunction = LogisticExciteFunction()
    else:
      raise NotImplementedError('unknown excite function type'+exciteFunctionType)

    if decayFunctionType == 'NoDecay':
      self._decayFunction = NoDecayFunction()
    elif decayFunctionType == 'Exponential':
      self._decayFunction = ExponentialDecayFunction(decayTimeConst)
    else:
      raise NotImplementedError('unknown decay function type'+decayFunctionType)


    # The maximum number of cells allowed in a single union SDR
    self._maxUnionCells = int(self._numColumns * self._maxUnionActivity)

    # Scalar activation of potential union SDR cells; most active cells become
    # the union SDR
    self._poolingActivation = numpy.zeros(self._numColumns, dtype=REAL_DTYPE)

    # include a small amount of tie-breaker when sorting pooling activation
    numpy.random.seed(1)
    self._poolingActivation_tieBreaker = numpy.random.randn(self._numColumns) * _TIE_BREAKER_FACTOR

    # time since last pooling activation increment
    # initialized to be a large number
    self._poolingTimer = numpy.ones(self._numColumns, dtype=REAL_DTYPE) * 1000

    # pooling activation level after the latest update, used for sigmoid decay function
    self._poolingActivationInitLevel = numpy.zeros(self._numColumns, dtype=REAL_DTYPE)

    # Current union SDR; the output of the union pooler algorithm
    self._unionSDR = numpy.array([], dtype=UINT_DTYPE)

    # Indices of active cells from spatial pooler
    self._activeCells = numpy.array([], dtype=UINT_DTYPE)

    # lowest possible pooling activation level
    self._poolingActivationlowerBound = 0.1

    self._preActiveInput = numpy.zeros(self._numInputs, dtype=REAL_DTYPE)
    # predicted inputs from the last n steps
    self._preLearningCandidates = numpy.zeros((self._numInputs, self._historyLength), dtype=REAL_DTYPE)

    if useInternalLateralConnections:
      self.internalDistalPermanences = SparseMatrix(self._numColumns, self._numColumns)
    self.distalPermanences = tuple(SparseMatrix(self._numColumns, n)
                                   for n in lateralInputWidths)



  def reset(self):
    """
    Reset the state of the Union Temporal Pooler.
    """

    # Reset Union Temporal Pooler fields
    self._poolingActivation = numpy.zeros(self._numColumns, dtype=REAL_DTYPE)
    self._unionSDR = numpy.array([], dtype=UINT_DTYPE)
    self._poolingTimer = numpy.ones(self._numColumns, dtype=REAL_DTYPE) * 1000
    self._poolingActivationInitLevel = numpy.zeros(self._numColumns, dtype=REAL_DTYPE)
    self._preActiveInput = numpy.zeros(self._numInputs, dtype=REAL_DTYPE)
    self._preLearningCandidates = numpy.zeros((self._numInputs, self._historyLength), dtype=REAL_DTYPE)

    # Reset Spatial Pooler fields
    self.setOverlapDutyCycles(numpy.zeros(self._numColumns, dtype=REAL_DTYPE))
    self.setActiveDutyCycles(numpy.zeros(self._numColumns, dtype=REAL_DTYPE))
    self.setMinOverlapDutyCycles(numpy.zeros(self._numColumns, dtype=REAL_DTYPE))
    self.setMinActiveDutyCycles(numpy.zeros(self._numColumns, dtype=REAL_DTYPE))
    self.setBoostFactors(numpy.ones(self._numColumns, dtype=REAL_DTYPE))


  def compute(self, activeInput, predictedActiveInput, learn,
              predictedCells = None, winnerCells = None, lateralInputs = (),):
    """
    Computes one cycle of the Union Temporal Pooler algorithm.
    @param activeInput            (numpy array) A numpy array of 0's and 1's that comprises the input to the union pooler
    @param predictedActiveInput   (numpy array) A numpy array of 0's and 1's that comprises the correctly predicted input to the union pooler
    @param learn                  (boolean)      A boolean value indicating whether learning should be performed
    """
    assert numpy.size(activeInput) == self._numInputs
    assert numpy.size(predictedActiveInput) == self._numInputs
    self.spatialPooler._updateBookeepingVars(learn)
    self.count += 1

    prevActiveCells = copy.deepcopy(self._unionSDR)


    overlapsPredictedActive = self.spatialPooler._calculateOverlap(predictedActiveInput)
    # Compute proximal dendrite overlaps with active and active-predicted inputs
    if predictedCells is not None and len(predictedCells) > 30:
      self._poolingTimer += 2

    overlapsActive = self.spatialPooler._calculateOverlap(activeInput)
    totalOverlap = (overlapsActive * self._activeOverlapWeight +
                    overlapsPredictedActive *
                    self._predictedActiveOverlapWeight).astype(REAL_DTYPE)

    if learn:
      boostFactors = numpy.zeros(self._numColumns, dtype=REAL_DTYPE)
      self.spatialPooler.getBoostFactors(boostFactors)
      boostedOverlaps = boostFactors * totalOverlap
    else:
      boostedOverlaps = totalOverlap

    segmentMultipliers = self._computeDistal(lateralInputs, self._unionSDR)
    multipliedOverlaps = segmentMultipliers * boostedOverlaps

    targetToActivate = max(self._maxUnionCells - len(self._unionSDR), self._numActive)
    activeCells = self._fuzzyInhibitColumnsGlobal(multipliedOverlaps,
                                                  targetToActivate,)
                                                  #self._inhibitionFactor)
    self._activeCells = activeCells

    # Decrement pooling activation of all cells
    self._decayPoolingActivation()

    # Update the poolingActivation of current active Union Temporal Pooler cells
    self._addToPoolingActivation(activeCells, overlapsPredictedActive)

    # update union SDR
    self._fuzzyGetMostActiveCells()#self._inhibitionFactor)

    #if winnerCells is not None:
    #  learningCandidates = winnerCells
    # else:
    learningCandidates = predictedActiveInput

    learningCells = numpy.intersect1d(activeCells, self._unionSDR)
    if learn:
      # Adapt permanence of connections to the currently active cells.
      self._adaptSynapses(learningCandidates, learningCells, self._synPermActiveInc, self._synPermInactiveDec)

      # Learn on whatever lateral connections there might be.
      if self.useInternalLateralConnections:
        self._learn(self.internalDistalPermanences, self._random,
                    learningCells, prevActiveCells, prevActiveCells,
                    self.sampleSizeDistal, self.initialDistalPermanence,
                    self.synPermDistalInc, self.synPermDistalDec,
                    self.connectedPermanenceDistal)

      for i, lateralInput in enumerate(lateralInputs):
        self._learn(self.distalPermanences[i], self._random,
                    learningCells, lateralInput, lateralInput,
                    self.sampleSizeDistal, self.initialDistalPermanence,
                    self.synPermDistalInc, self.synPermDistalDec,
                    self.connectedPermanenceDistal)

      # Increase permanence of connections from predicted active inputs to cells in the union SDR
      # This is Hebbian learning applied to the current time step
      if self._synPermPredActiveInc > 0:
        self._adaptSynapses(learningCandidates, self._unionSDR, self._synPermPredActiveInc, 0.0)

      # adapt permenence of connections from previously predicted inputs to newly active cells
      # This is a reinforcement learning rule that considers previous input to the current cell
      if self._synPermPreviousPredActiveInc > 0:
        for i in xrange(self._historyLength):
          self._adaptSynapses(self._preLearningCandidates[:,i], activeCells, self._synPermPreviousPredActiveInc, 0.0)

      # Homeostasis learning inherited from the spatial pooler
      self.spatialPooler._updateDutyCycles(totalOverlap.astype(UINT_DTYPE), activeCells)
      self.spatialPooler._bumpUpWeakColumns()
      self.spatialPooler._updateBoostFactors()
      if self.spatialPooler._isUpdateRound():
        self.spatialPooler._updateInhibitionRadius()
        self.spatialPooler._updateMinDutyCycles()

    # save inputs from the previous time step
    self._preActiveInput = copy.copy(activeInput)
    self._preLearningCandidates = numpy.roll(self._preLearningCandidates,1,1)
    if self._historyLength > 0:
      self._preLearningCandidates[:, 0] = learningCandidates

    return self._unionSDR


  def _computeDistal(self, lateralInputs, prevActiveCells):
    """
    Computes overall impact of distal segments for each cell.  Returns a vector
    of distal multipliers, which can be directly multiplied with overlap scores
    """
    # Calculate the number of active segments on each cell
    numActiveSegmentsByCell = numpy.zeros(self._numColumns, dtype="int")
    if self.useInternalLateralConnections:
      overlaps = self.internalDistalPermanences.rightVecSumAtNZGteThresholdSparse(
        prevActiveCells, self.connectedPermanenceDistal)
      numActiveSegmentsByCell[overlaps >= self.activationThresholdDistal] += 1
    for i, lateralInput in enumerate(lateralInputs):
      overlaps = self.distalPermanences[i].rightVecSumAtNZGteThresholdSparse(
        lateralInput, self.connectedPermanenceDistal)
      numActiveSegmentsByCell[overlaps >= self.activationThresholdDistal] += 1

    distalMultipliers = numpy.full(self._numColumns, self.segmentBoost)
    return distalMultipliers ** numActiveSegmentsByCell


  @staticmethod
  def _learn(# mutated args
             permanences, rng,

             # activity
             activeCells, activeInput, growthCandidateInput,

             # configuration
             sampleSize, initialPermanence, permanenceIncrement,
             permanenceDecrement, connectedPermanence):
    """
    For each active cell, reinforce active synapses, punish inactive synapses,
    and grow new synapses to a subset of the active input bits that the cell
    isn't already connected to.  This only covers distal and apical learning
    for the Union Temporal Pooler, as proximal learning is handled via the
    Spatial Pooler mechanics and does not include segments.

    Parameters:
    ----------------------------
    @param  permanences (SparseMatrix)
            Matrix of permanences, with cells as rows and inputs as columns

    @param  rng (Random)
            Random number generator

    @param  activeCells (sorted sequence)
            Sorted list of the cells that are learning

    @param  activeInput (sorted sequence)
            Sorted list of active bits in the input

    @param  growthCandidateInput (sorted sequence)
            Sorted list of active bits in the input that the activeCells may
            grow new synapses to

    For remaining parameters, see the __init__ docstring.
    """

    permanences.incrementNonZerosOnOuter(
      activeCells, activeInput, permanenceIncrement)
    permanences.incrementNonZerosOnRowsExcludingCols(
      activeCells, activeInput, -permanenceDecrement)
    permanences.clipRowsBelowAndAbove(
      activeCells, 0.0, 1.0)
    if sampleSize == -1:
      permanences.setZerosOnOuter(
        activeCells, activeInput, initialPermanence)
    else:
      existingSynapseCounts = permanences.nNonZerosPerRowOnCols(
        activeCells, activeInput)

      maxNewByCell = numpy.empty(len(activeCells), dtype="int32")
      numpy.subtract(sampleSize, existingSynapseCounts, out=maxNewByCell)

      permanences.setRandomZerosOnOuter(
        activeCells, growthCandidateInput, maxNewByCell, initialPermanence, rng)



  def _fuzzyInhibitColumnsGlobal(self, overlaps, numActive):
    """
    Perform global inhibition. Performing global inhibition entails picking the
    top 'numActive' columns with the highest overlap score in the entire
    region, and all cells whose overlaps are within InhibitionFactor of the
    average overlap score of the top numActive cells.

    :param overlaps: an array containing the overlap score for each  column.
                    The overlap score for a column is defined as the number
                    of synapses in a "connected state" (connected synapses)
                    that are connected to input bits which are turned on.
    :param inhibitionFactor: a float in [0, 1], which specifies how strongly
                    columns inhibit each other.  At 0 all columns will become
                    active at all time steps, while at 1 behavior will be
                    almost identical to that of the normal inibition function.
    :param numActive: The desired minimum number of active bits.
    @return list with indices of the winning columns
    """
    # Calculate winners using stable sort algorithm (mergesort)
    # for compatibility with C++
    sortedWinnerIndices = numpy.argsort(overlaps, kind='mergesort')

    # Calculate the inhibition threshold
    start = int(len(sortedWinnerIndices) - numActive)
    winners = sortedWinnerIndices[start:]
    if len(winners) == 0:
      return numpy.asarray([], dtype = "uint32")

    threshold = numpy.mean(overlaps[winners])*self._inhibitionFactor

    # Determine which other cells will become active
    while start > 0:
      i = sortedWinnerIndices[start]
      if overlaps[i] <= threshold:
        break
      else:
        start -= 1

    return sortedWinnerIndices[start:][::-1]


  def _decayPoolingActivation(self):
    """
    Decrements pooling activation of all cells
    """
    if self._decayFunctionType == 'NoDecay':
      self._poolingActivation = self._decayFunction.decay(self._poolingActivation)
    elif self._decayFunctionType == 'Exponential':
      self._poolingActivation = self._decayFunction.decay(\
                                self._poolingActivationInitLevel, self._poolingTimer)

    return self._poolingActivation


  def _addToPoolingActivation(self, activeCells, overlaps):
    """
    Adds overlaps from specified active cells to cells' pooling
    activation.
    @param activeCells: Indices of those cells winning the inhibition step
    @param overlaps: A current set of overlap values for each cell
    @return current pooling activation
    """
    self._poolingActivation[activeCells] = self._exciteFunction.excite(
      self._poolingActivation[activeCells], overlaps[activeCells])

    # increase pooling timers for all cells
    self._poolingTimer[self._poolingTimer >= 0] += 1

    # reset pooling timer for active cells
    self._poolingTimer[activeCells] = 1
    self._poolingActivationInitLevel[activeCells] = self._poolingActivation[activeCells]

    return self._poolingActivation


  def _getActiveCells(self):
    return self._activeCells


  def _getMostActiveCells(self):
    """
    Gets the most active cells in the Union SDR having at least non-zero
    activation in sorted order.
    @return: a list of cell indices
    """
    poolingActivation = self._poolingActivation
    nonZeroCells = numpy.argwhere(poolingActivation > 0)[:,0]

    # include a tie-breaker before sorting
    poolingActivationSubset = poolingActivation[nonZeroCells] + \
                              self._poolingActivation_tieBreaker[nonZeroCells]
    potentialUnionSDR = nonZeroCells[numpy.argsort(poolingActivationSubset)[::-1]]

    topCells = potentialUnionSDR[0: self._maxUnionCells]

    if max(self._poolingTimer) > self._minHistory:
      self._unionSDR = numpy.sort(topCells).astype(UINT_DTYPE)
    else:
      self._unionSDR = []

    return self._unionSDR


  def _fuzzyGetMostActiveCells(self):
    """
    Gets the most active cells in the Union SDR having at least non-zero
    activation in sorted order.  However, uses fuzzy inhibition, so cells with
    activations close to the threshold will also be included.
    @return: a list of cell indices
    """
    poolingActivation = self._poolingActivation
    nonZeroCells = numpy.argwhere(poolingActivation > 0)[:,0]

    poolingActivationSubset = poolingActivation[nonZeroCells]
    sortedActivations = numpy.argsort(poolingActivationSubset)[::-1]
    potentialUnionSDR = nonZeroCells[sortedActivations]

    start = self._maxUnionCells
    winners = sortedActivations[:start]
    threshold = numpy.mean(poolingActivationSubset[winners])*self._inhibitionFactor

    # Determine which other cells will become active
    while start < len(nonZeroCells):
      if poolingActivationSubset[sortedActivations[start]] <= threshold:
        break
      else:
        start += 1

    topCells = potentialUnionSDR[0: start]

    if max(self._poolingTimer) > self._minHistory:
      self._unionSDR = numpy.sort(topCells).astype(UINT_DTYPE)
    else:
      self._unionSDR = []

    return self._unionSDR


  # override
  def _adaptSynapses(self, inputVector, activeColumns, synPermActiveInc, synPermInactiveDec):
    """
    The primary method in charge of learning. Adapts the permanence values of
    the synapses based on the input vector, and the chosen columns after
    inhibition round. Permanence values are increased for synapses connected to
    input bits that are turned on, and decreased for synapses connected to
    inputs bits that are turned off.

    Parameters:
    ----------------------------
    @param inputVector:
                    A numpy array of 0's and 1's that comprises the input to
                    the spatial pooler. There exists an entry in the array
                    for every input bit.
    @param activeColumns:
                    An array containing the indices of the columns that
                    survived inhibition.

    @param synPermActiveInc:
                    Permanence increment for active inputs
    @param synPermInactiveDec:
                    Permanence decrement for inactive inputs
    """
    inputIndices = numpy.where(inputVector > 0)[0]
    permChanges = numpy.zeros(self._numInputs, dtype=REAL_DTYPE)
    permChanges.fill(-1 * synPermInactiveDec)
    permChanges[inputIndices] = synPermActiveInc
    perm = numpy.zeros(self._numInputs, dtype=REAL_DTYPE)
    potential = numpy.zeros(self._numInputs, dtype=REAL_DTYPE)
    for i in activeColumns:
      i = numpy.uint(i)
      self.getPermanence(i, perm)
      self.getPotential(i, potential)
      maskPotential = numpy.where(potential > 0)[0]
      perm[maskPotential] += permChanges[maskPotential]
      self._updatePermanencesForColumn(perm, i, False)


  def getUnionSDR(self):
    return self._unionSDR
