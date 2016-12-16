package classification;

import java.util.ArrayList;
import java.util.Collections;

/**
 * Created by nacos on 11/16/2016.
 */
public class RandomForest extends Classifier{

    public static void main(String[] args) throws Exception{
        /* Handle arguments */
        if(args.length != 2)
            throw new IllegalArgumentException("Arguments should contain a train-file and a test-file!");

        RandomForest randomForest = new RandomForest(args[0], args[1], 100);

        randomForest.readDataFromFiles();

        randomForest.train();

        //randomForest.printRFStructure();

        randomForest.evaluateQuality();

        final int[][] matrix = randomForest.getConfusionMatrix();
        int k = randomForest.getLabelOptions();
        for(int i = 0; i < k; i++){
            for(int j = 0; j < k; j++){
                System.out.print(matrix[i][j] + "\t");
            }
            System.out.println();
        }
        return;
    }

    // Configuration
    private int _forestSize;

    // Classifier
    private ArrayList<DecisionTreeKernel> _decisionTreeKernels;

    /**
     * Constructor
     */
    public RandomForest(final String trainFileName, final String testFileName, final int forestSize){
        super(trainFileName, testFileName);

        if(forestSize <= 0){
            throw new IllegalArgumentException("Forest size must be positive!");
        }
        this._forestSize = forestSize;
    }

    /**
     * Step 3. Implement Ensemble Classification Method
     * Train a random forest from training data.
     */
    @Override
    public int train() throws Exception{
        if(_trainSampleSize <= 0){
            throw new Exception("Please read valid train and test data before training random forest!");
        }

        /* Seperately train each decision tree kernel */
        _decisionTreeKernels = new ArrayList<>(_forestSize);

        // Training data's indexes to shuffle
        ArrayList<Integer> trainSampleIndexesToShuffle = new ArrayList<>(_trainSampleSize);
        for(int j = 0; j < _trainSampleSize; j++)
            trainSampleIndexesToShuffle.add(j);

        for(int i = 0; i < _forestSize; i++){
            /* TODO: Make a bootstrap of the training samples */
            ArrayList<Integer> trainLabels = new ArrayList<>(_trainSampleSize);
            ArrayList<int[]> trainAttrs = new ArrayList<>(_trainSampleSize);

            // Shuffle the training data's indexes
            Collections.shuffle(trainSampleIndexesToShuffle);

            // Re-organize the training sample according to shuffled indexes
            for(int j = 0; j < _trainSampleSize; j++){
                int sampleIndex = trainSampleIndexesToShuffle.get(j);
                trainLabels.add(_trainLabels.get(sampleIndex));
                trainAttrs.add(_trainAttrs.get(sampleIndex));
            }

            DecisionTreeKernel decisionTreeKernel = new DecisionTreeKernel(true,
                    _labelOptions, _attrCount, _attrOptions,
                    _trainSampleSize, trainLabels, trainAttrs);
            decisionTreeKernel.train();

            _decisionTreeKernels.add(decisionTreeKernel);
        }
        return 0;
    }

    /**
     * Use the trained decision tree to predict a sample's label
     */
    @Override
    public int predictSample(final int[] sampleAttr){
        /* Check if random forest not grown */
        if(_decisionTreeKernels.size() == 0){
            System.err.println("Random Forest must be trained before it can predict any sample!");
            return -1;
        }

        /* Record every decision tree's prediction */
        int[] candidates = new int[_labelOptions];
        for(int i = 0; i < _forestSize; i++){
            int prediction = _decisionTreeKernels.get(i).predictSample(sampleAttr);
            candidates[prediction - 1]++;
        }

        /* Select the prediction with most votes */
        int maxVotes = 0;
        int maxVoteIndex = 0;
        for(int i = 0; i < _labelOptions; i++){
            if(candidates[i] > maxVotes){
                maxVoteIndex = i;
                maxVotes = candidates[i];
            }
        }

        return maxVoteIndex + 1;
    }

    public void printRFStructure(){
        for(int i = 0; i < _forestSize; i++){
            System.out.println("=== DT " + i + " ===");
            _decisionTreeKernels.get(i).printDTStructure();
            System.out.println();
        }
    }
}
