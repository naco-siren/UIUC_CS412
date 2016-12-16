package classification;

import java.io.*;
import java.util.ArrayList;

/**
 * Created by nacos on 11/16/2016.
 */
public class Classifier {
    // Configuration
    protected String _trainFilename;
    protected String _testFilename;

    // Meta-data
    protected int _labelOptions;
    protected int _attrCount;
    protected int[] _attrOptions;

    // Train Data
    protected int _trainSampleSize;
    protected ArrayList<Integer> _trainLabels;
    protected ArrayList<int[]> _trainAttrs;

    // Test Data
    protected int _testSampleSize;
    protected ArrayList<Integer> _testLabels;
    protected ArrayList<int[]> _testAttrs;

    // Evaluation
    protected int[][] _confusionMatrix;

    /**
     * Constructor
     */
    protected Classifier(final String trainFileName, final String testFileName){
        this._trainFilename = trainFileName;
        this._testFilename = testFileName;
    }

    /**
     * Step 1: Data I/O and Data Format.
     * Read training and test data from file.
     */
    protected int readDataFromFiles() throws Exception{
        readTrainData();
        readTestData();
        return 0;
    }
    /**
     * Step 1: Data I/O and Data Format.
     * Read training data from file.
     */
    protected int readTrainData() throws Exception{
        /* Use a reader for train-file input */
        File trainFile = new File(_trainFilename);
        if(trainFile.exists() == false || trainFile.isFile() == false)
            throw new FileNotFoundException("Train file not found!");

        BufferedReader bufferedReader = new BufferedReader(
                new InputStreamReader(
                        new FileInputStream(trainFile)
                )
        );


        /* Iterate whole data set the first time to grasp the data's dimension of label and attr */
        _labelOptions = 0;
        _attrCount = 0;

        ArrayList<String> trainDataLines = new ArrayList<>();
        String line;
        while((line = bufferedReader.readLine()) != null){
            if(line.length() == 0)
                continue;

            String words[] = line.split(" ");

            // Label type
            int label = Integer.parseInt(words[0]);
            if(label > _labelOptions)
                _labelOptions = label;

            // Attribute count
            int maxAttr = Integer.parseInt(words[words.length-1].split(":")[0]);
            if(maxAttr > _attrCount)
                _attrCount = maxAttr;

            trainDataLines.add(line);
        }
        _trainSampleSize = trainDataLines.size();

        /* Prepare data to store samples */
        if(_labelOptions < 2 || _attrCount < 1)
            throw new Exception("Train data seems to be corrupted!");

        // Metadata
        _attrOptions = new int[_attrCount];
        // Data
        _trainLabels = new ArrayList<>(_trainSampleSize);
        _trainAttrs = new ArrayList<>(_trainSampleSize);

        /* Iterate whole data set the second time to save data into array and specify each attr's options */
        for (int i = 0; i < _trainSampleSize; i++) {
            String words[] = trainDataLines.get(i).split(" ");

            // Label
            _trainLabels.add(Integer.parseInt(words[0]));

            // Attributes
            int[] thisSampleAttrs = new int[_attrCount];
            for(int j = 1; j < words.length; j++){
                String[] pair = words[j].split(":");
                if(pair.length != 2)
                    throw new Exception("Data corruption!");

                int attrIndex = Integer.parseInt(pair[0]);
                int attrValue = Integer.parseInt(pair[1]);
                thisSampleAttrs[attrIndex-1] = attrValue;

                if(attrValue > _attrOptions[attrIndex-1])
                    _attrOptions[attrIndex-1] = attrValue;
            }
            _trainAttrs.add(thisSampleAttrs);
        }

        for(int i = 0; i < _attrCount; i++){
            _attrOptions[i]++;
        }

        bufferedReader.close();
        return 0;
    }

    /**
     * Step 1: Data I/O and Data Format.
     * Read test data from file.
     */
    protected int readTestData() throws Exception{
        /* Use a reader for train-file input */
        File testFile = new File(_testFilename);
        if(testFile.exists() == false || testFile.isFile() == false)
            throw new FileNotFoundException("Test file not found!");

        BufferedReader bufferedReader = new BufferedReader(
                new InputStreamReader(
                        new FileInputStream(testFile)
                )
        );

        /* Iterate whole data set the first time to grasp the data's dimension of label and attr */
        int labelOptions = 0;
        int attrCount = 0;

        ArrayList<String> testDataLines = new ArrayList<>();
        String line;
        while((line = bufferedReader.readLine()) != null){
            if(line.length() == 0)
                continue;

            String words[] = line.split(" ");

            // Label type
            int label = Integer.parseInt(words[0]);
            if(label > labelOptions)
                labelOptions = label;

            // Attribute count
            int maxAttr = Integer.parseInt(words[words.length-1].split(":")[0]);
            if(maxAttr > attrCount)
                attrCount = maxAttr;

            testDataLines.add(line);
        }
        _testSampleSize = testDataLines.size();

        /* Prepare data to store samples */
        if(labelOptions < 2 || attrCount < 1)
            throw new Exception("Test data's dimension is too small!");
        if(labelOptions > _labelOptions || attrCount > _attrCount)
            throw new Exception("Test data has larger dimension than train data!");

        // Metadata
        int[] attrOptions = new int[_attrCount];
        // Data
        _testLabels = new ArrayList<>(_testSampleSize);
        _testAttrs = new ArrayList<>(_testSampleSize);

        /* Iterate whole data set the second time to save data into array and specify each attr's options */
        for (int i = 0; i < _testSampleSize; i++) {
            String words[] = testDataLines.get(i).split(" ");

            // Label
            _testLabels.add(Integer.parseInt(words[0]));

            // Attributes
            int[] thisSampleAttrs = new int[_attrCount];
            for(int j = 1; j < words.length; j++){
                String[] pair = words[j].split(":");
                if(pair.length != 2)
                    throw new Exception("Data corruption!");

                int attrIndex = Integer.parseInt(pair[0]);
                int attrValue = Integer.parseInt(pair[1]);
                thisSampleAttrs[attrIndex-1] = attrValue;

                if(attrValue > attrOptions[attrIndex-1])
                    attrOptions[attrIndex-1] = attrValue;
            }
            _testAttrs.add(thisSampleAttrs);
        }

        for(int i = 0; i < _attrCount; i++){
            attrOptions[i]++;
        }

        bufferedReader.close();
        return 0;
    }

    /**
     * Step 2:Implement Basic Classification Method.
     * Train a decision tree from training data.
     */
    protected int train() throws Exception{
        throw new Exception("To be implement!");
    }

    /**
     * Step 2:Implement Basic Classification Method.
     * Use the trained decision tree to predict a sample's label
     */
    protected int predictSample(final int[] sampleAttr) throws Exception{
        throw new Exception("To be implement!");
    }

    /**
     * Generate confusion matrix on the test data.
     */
    public int evaluateQuality() throws Exception{
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
    protected int getLabelOptions() {
        return _labelOptions;
    }
    protected int getAttrCount() {
        return _attrCount;
    }
    protected int[] getAttrOptions() {
        return _attrOptions;
    }
    protected int[][] getConfusionMatrix() {
        return _confusionMatrix;
    }
}
