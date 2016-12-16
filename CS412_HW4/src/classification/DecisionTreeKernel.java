package classification;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

/**
 * Created by nacos on 11/4/2016.
 */
public class DecisionTreeKernel {
    // Configuration
    private boolean _isForestRI;

    // Metadata
    private int _labelOptions;
    private int _attrCount;
    private int[] _attrOptions;

    // Train Data
    private int _trainSampleSize;
    private ArrayList<Integer> _trainLabels;
    private ArrayList<int[]> _trainAttrs;

    // Classifier
    private DTNode _DTRoot;

    /**
     * Constructor
     */
    public DecisionTreeKernel(final boolean isForestRI,
                              int labelOptions, int attrCount, int[] attrOptions,
                              int trainSampleSize, ArrayList<Integer> trainLabels, ArrayList<int[]> trainAttrs){
        // Configuration
        this._isForestRI = isForestRI;

        // Metadata
        this._labelOptions = labelOptions;
        this._attrCount = attrCount;
        this._attrOptions = attrOptions;

        // Train Data
        this._trainSampleSize = trainSampleSize;
        this._trainLabels = trainLabels;
        this._trainAttrs = trainAttrs;
    }

    /**
     * Step 2:Implement Basic Classification Method.
     * Train a decision tree from training data.
     */
    public int train() throws Exception{
        /* Check if data OK */
        if((_trainLabels != null && _trainSampleSize > 0) == false){
            System.err.println("Please read valid train and test data before training decision tree!");
            return -1;
        }

        /* Initiate attributes usage status */
        boolean[] availableAttrs = new boolean[_attrCount];
        for(int i = 0; i < _attrCount; i++)
            availableAttrs[i] = true;

        /* Instantiate a DTNode as DecisionTree's root and grow its branches */
        _DTRoot = new DTNode(_trainLabels, _trainAttrs, 0, -1, -1, -1, _attrCount, availableAttrs);
        _DTRoot.growBranch();

        return 0;
    }

    /**
     * Step 2:Implement Basic Classification Method.
     * Use the trained decision tree to predict a sample's label
     */
    public int predictSample(final int[] sampleAttr){
        /* Check if tree not grown */
        if(_DTRoot == null){
            System.err.println("Decision tree must be trained before it can predict any sample!");
            return -1;
        }

        /* Trace along the decision tree */
        DTNode node = _DTRoot;
        while(node._thisAttrIndex != -1){
            node = node._childrenNodes[sampleAttr[node._thisAttrIndex]];
        }
        return node._predictLabel;
    }

    /**
     * Step 2:Implement Basic Classification Method.
     * Print the decision tree to standard output.
     */
    public void printDTStructure(){
        boolean[] omits = new boolean[_attrCount];
        _DTRoot.printBranch(omits);
    }


    /**
     * Calculate the Gini-Index of a sample dataset on a given attr by building a AVC-list.
     * @param attrIndex specifies the given attribute index.
     */
    public double getGiniIndexOnAttr(final ArrayList<Integer> sampleLabels, final ArrayList<int[]> sampleAttrs, final int attrIndex){
        double sampleSize = sampleLabels.size();
        int thisAttrOptions = _attrOptions[attrIndex];

        /* Build AVC-set */
        int[][] AVCset = new int[thisAttrOptions][_labelOptions];
        for(int i = 0; i < sampleSize; i++){
            int sampleLabel = sampleLabels.get(i);
            int sampleThisAttr = sampleAttrs.get(i)[attrIndex];
            AVCset[sampleThisAttr][sampleLabel-1]++;
        }

        /* Calculate Gini Index for each value of this attribute */
        double[] attrValueSums = new double[thisAttrOptions];
        double[] attrValueGini = new double[thisAttrOptions];
        for(int i = 0; i < thisAttrOptions; i++){
            attrValueGini[i] = 1;
            for(int j = 0; j < _labelOptions; j++){
                attrValueSums[i] += AVCset[i][j];
            }
        }
        for(int i = 0; i < thisAttrOptions; i++){
            for(int j = 0; j < _labelOptions; j++) {
                if(attrValueSums[i] > 0)
                    attrValueGini[i] -= Math.pow(AVCset[i][j] / attrValueSums[i], 2);
            }
        }

        /* Sum up each value's Gini Index */
        double giniIndex = 0;
        for(int i = 0; i < thisAttrOptions; i++){
            giniIndex += (attrValueGini[i] * attrValueSums[i] / sampleSize);
        }
        return giniIndex;
    }

    /**
     * Calculate the Gini-Index of a sample dataset by building a AVC-list.
     */
    public double getGiniIndex(final ArrayList<Integer> sampleLabels, final ArrayList<int[]> sampleAttrs){
        double sampleSize = sampleLabels.size();

        int[] AVCset = new int[_labelOptions];

        for(int i = 0; i < sampleSize; i++){
            int sampleLabel = sampleLabels.get(i);
            AVCset[sampleLabel-1] ++;
        }

        /* Calculate Gini Index for this sample data */
        double giniIndex = 1;
        for(int j = 0; j < _labelOptions; j++) {
            giniIndex -= Math.pow(AVCset[j] / sampleSize, 2);
        }
        return giniIndex;
    }

    /*
     * A class representing the nodes in a decision tree
     */
    public class DTNode {
        // Sample data
        private ArrayList<Integer> _sampleLabels;
        private ArrayList<int[]> _sampleAttrs;
        private int _thisAttrOptions;

        // Tree-growth status
        private int _remainingAttrsCount;
        private boolean[] _availableAttrs;

        // Current node data
        private int _prevAttrIndex;
        private int _prevAttrValue;
        private int _thisAttrIndex;
        private int _predictLabel;
        private int _parentPopularLabel;

        // Children nodes data
        private DTNode[] _childrenNodes;
        private int _branchDepth;
        private int _nodeLayer;

        /**
         * Public constructor of DecisionTree Node
         * @param sampleLabels  Labels of the sample data classified into this node.
         * @param sampleAttrs   Attributes of the sample data classified into this node.
         * @param prevAttrIndex The attribute that this node is grown on .
         * @param prevAttrValue This node's value on the attribute.
         * @param remainingAttrsCount   Available attributes left for further partitioning.
         * @param availableAttrs    Boolean array to indicate available attributes left.
         */
        private DTNode(final ArrayList<Integer> sampleLabels, final ArrayList<int[]> sampleAttrs,
                       final int parentLayer, final int prevAttrIndex, final int prevAttrValue, final int parentPopularLabel,
                       final int remainingAttrsCount, final boolean[] availableAttrs) {
            this._sampleLabels = sampleLabels;
            this._sampleAttrs = sampleAttrs;

            this._prevAttrIndex = prevAttrIndex;
            this._prevAttrValue = prevAttrValue;
            this._thisAttrIndex = -1;
            this._predictLabel = -1;
            this._parentPopularLabel = parentPopularLabel;

            this._remainingAttrsCount = remainingAttrsCount;
            this._availableAttrs = availableAttrs;

            this._nodeLayer = parentLayer;
        }

        private int growBranch() {
            int sampleSize = _sampleLabels.size();

            /* Find the most possible value */
            int[] possibleLabels = new int[_labelOptions];
            for(int i = 0; i < sampleSize; i++) {
                int thisSampleLabel = _sampleLabels.get(i);
                possibleLabels[thisSampleLabel - 1]++;
            }
            int mostPossibleLabel = -1;
            int mostPossibleLabelCount = 0;
            for(int j = 0; j < _labelOptions; j++){
                if(possibleLabels[j] > mostPossibleLabelCount){
                    mostPossibleLabelCount = possibleLabels[j];
                    mostPossibleLabel = j + 1;
                }
            }


            /* Decide whether or not to stop partitioning */
            boolean shouldStop = false;
            if(_remainingAttrsCount == 0){ // No remaining attributes for further partitioning.
                shouldStop = true;
            } else if(sampleSize == 0){ // No samples left.
                shouldStop = true;
            } else if(mostPossibleLabelCount == sampleSize){ // All samples belong to the same class
                shouldStop = true;
            }
            if(shouldStop == true) {
                if(mostPossibleLabel != -1) {
                    _predictLabel = mostPossibleLabel;
                } else {
                    _predictLabel = _parentPopularLabel;
                }
                _branchDepth = 0;
                return _branchDepth;
            }


            /* Decide on a partition attribute */
            double giniIndex = getGiniIndex(_sampleLabels, _sampleAttrs);
            _thisAttrIndex = -1;
            double maxGiniIndex = 0;
            double[] reductionInImpurities = new double[_attrCount];

            // TODO: Bootstrap
            boolean[] availableAttrs = _availableAttrs.clone();
            if(_isForestRI == true){
                // Collect all candidates
                ArrayList<Integer> candidateAttrIndexes = new ArrayList<>();
                for(int i = 0; i < _attrCount; i++){
                    if(availableAttrs[i] == true){
                        candidateAttrIndexes.add(i);
                    }
                }

                // Shuffle and select top k (sqrt());
                int candidateAttrsCount = (int) Math.sqrt(_remainingAttrsCount);
                Collections.shuffle(candidateAttrIndexes);
                for(int i = candidateAttrsCount; i < candidateAttrIndexes.size(); i++){
                    availableAttrs[candidateAttrIndexes.get(i)] = false;
                }
                _thisAttrIndex = candidateAttrIndexes.get(0);
            }

            // Find the attribute with the maximum Gini index to split
            for(int i = 0; i < _attrCount; i++){
                if(availableAttrs[i] == false)
                    continue;

                /* Select current attribute if it has the max reduction in impurity */
                reductionInImpurities[i] = giniIndex - getGiniIndexOnAttr(_sampleLabels, _sampleAttrs, i);
                if(reductionInImpurities[i] >= maxGiniIndex){
                    _thisAttrIndex = i;
                    maxGiniIndex = reductionInImpurities[i];
                }
            }
            _thisAttrOptions = _attrOptions[_thisAttrIndex];


            /* Partition the samples by the decided attribute */
            ArrayList<ArrayList<Integer>> childrenLabels = new ArrayList<>(_thisAttrOptions);
            ArrayList<ArrayList<int[]>> childrenAttrs = new ArrayList<>(_thisAttrOptions);
            for(int i = 0; i < _thisAttrOptions; i++){
                childrenLabels.add(new ArrayList<Integer>());
                childrenAttrs.add(new ArrayList<int[]>());
            }
            for(int i = 0; i < sampleSize; i++){
                int thisSamplePartitionAttrValue = _sampleAttrs.get(i)[_thisAttrIndex];
                childrenLabels.get(thisSamplePartitionAttrValue).add( _sampleLabels.get(i));
                childrenAttrs.get(thisSamplePartitionAttrValue).add(_sampleAttrs.get(i));
            }

            /* Grow a decision tree for each value's child */
            boolean[] newAvailableAttrs = _availableAttrs.clone();
            newAvailableAttrs[_thisAttrIndex] = false;

            _childrenNodes = new DTNode[_thisAttrOptions];
            Integer[] childrenDepths = new Integer[_thisAttrOptions];
            for(int i = 0; i < _thisAttrOptions; i++){
                _childrenNodes[i] = new DTNode(childrenLabels.get(i), childrenAttrs.get(i), _nodeLayer + 1, _thisAttrIndex, i, mostPossibleLabel, _remainingAttrsCount-1, newAvailableAttrs);
                childrenDepths[i] = _childrenNodes[i].growBranch();
            }

            /* Select the deepest branch then add 1 as this branch depth */
            _branchDepth = Collections.max(Arrays.asList(childrenDepths)) + 1;
            return _branchDepth;
        }

        private void printBranch(final boolean[] omits){
            if(_thisAttrIndex != -1) {
                System.out.println("\u2500\u2500 #" + _thisAttrIndex);
                for(int i = 0; i < _childrenNodes.length; i++){
                    /* Omit those vertical table edges according to @param omits */
                    for(int j = 0; j < _nodeLayer; j++) {
                        if(omits[j] == false)
                            System.out.print("    \u2502    ");
                        else
                            System.out.print("         ");
                    }

                    /* Decide the shape of the table corner before the last child
                       and the blank vertical table edges */
                    if(i == _childrenNodes.length - 1){
                        System.out.print("    \u2514\u2500 " + i + " ");
                        boolean[] nextOmits = omits.clone();
                        nextOmits[_nodeLayer] = true;
                        _childrenNodes[i].printBranch(nextOmits);
                    } else {
                        System.out.print("    \u251C\u2500 " + i + " ");
                        _childrenNodes[i].printBranch(omits);
                    }
                }
            } else {
                System.out.println("-> [" + _predictLabel + "]");
            }
        }
    }

    /**
     * Getters
     */
    public int getLabelOptions() {
        return _labelOptions;
    }
    public int getAttrCount() {
        return _attrCount;
    }
    public int[] getAttrOptions() {
        return _attrOptions;
    }
}
