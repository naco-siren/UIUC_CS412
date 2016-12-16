package classification;

import java.util.ArrayList;

/**
 * Created by nacos on 11/4/2016.
 */
public class DecisionTree extends Classifier{
    public static void main(String[] args) throws Exception{
        /* Handle arguments */
        if(args.length != 2)
            throw new IllegalArgumentException("Arguments should contain a train-file and a test-file!");

        DecisionTree decisionTree = new DecisionTree(args[0], args[1]);

        decisionTree.readDataFromFiles();

        decisionTree.train();

        //decisionTree.printDTStructure();

        decisionTree.evaluateQuality();

        final int[][] matrix = decisionTree.getConfusionMatrix();
        int k = decisionTree.getLabelOptions();
        for(int i = 0; i < k; i++){
            for(int j = 0; j < k; j++){
                System.out.print(matrix[i][j] + "\t");
            }
            System.out.println();
        }

        return;
    }

    // Configuration
    // NULL

    // Decision Tree
    private DecisionTreeKernel _decisionTreeKernel;


    /**
     * Constructor
     */
    public DecisionTree(final String trainFileName, final String testFileName){
        super(trainFileName, testFileName);
    }


    /**
     * Step 2:Implement Basic Classification Method.
     * Train a decision tree from training data.
     */
    @Override
    public int train() throws Exception{
        /* Check if data OK */
        if((_trainLabels != null && _trainSampleSize > 0 && _testLabels != null && _testSampleSize > 0 ) == false){
            System.err.println("Please read valid train and test data before training decision tree!");
            return -1;
        }

        /* Initiate attributes usage status */
        boolean[] availableAttrs = new boolean[_attrCount];
        for(int i = 0; i < _attrCount; i++)
            availableAttrs[i] = true;


        _decisionTreeKernel = new DecisionTreeKernel(false,
                    _labelOptions, _attrCount, _attrOptions,
                    _trainSampleSize, _trainLabels, _trainAttrs
                );
        _decisionTreeKernel.train();

        ArrayList<Integer> trainLabels = new ArrayList<>(_trainSampleSize);
        ArrayList<int[]> trainAttrs = new ArrayList<>(_trainSampleSize);

        return 0;
    }

    /**
     * Step 2:Implement Basic Classification Method.
     * Print the decision tree to standard output.
     */
    public void printDTStructure(){
        _decisionTreeKernel.printDTStructure();
    }

    /**
     * Step 2:Implement Basic Classification Method.
     * Use the trained decision tree to predict a sample's label
     */
    @Override
    public int predictSample(final int[] sampleAttr){
        /* Check if tree not grown */
        if(_decisionTreeKernel == null){
            System.err.println("Decision tree must be trained before it can predict any sample!");
            return -1;
        }

        return _decisionTreeKernel.predictSample(sampleAttr);
    }
    /**
     * Step 2:Implement Basic Classification Method.
     * Generate confusion matrix on the test data.
     */
    public int evaluateQuality(){
        _confusionMatrix = new int[_labelOptions][_labelOptions];

        for(int i = 0; i < _testSampleSize; i++){
            /* Compare predicted label to actual actual */
            int predictedLabel = predictSample(_testAttrs.get(i));
            int actualLabel = _testLabels.get(i);
            _confusionMatrix[actualLabel - 1][predictedLabel - 1]++;
        }

        return 0;
    }

    /**
     * Getters
     */
    public int[][] getConfusionMatrix() {
        return _confusionMatrix;
    }
}
