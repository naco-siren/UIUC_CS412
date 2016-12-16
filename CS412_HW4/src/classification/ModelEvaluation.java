package classification;

import java.text.DecimalFormat;

/**
 * Step 4. Model Evaluation and Report
 *
 * Created by nacos on 11/23/2016.
 */
public class ModelEvaluation {
    public static void main(String[] args) throws Exception{
        String[] datasetNames = new String[]{
                "balance",
                "nursery",
                "led",
                "poker"
        };
        String[] trainFileNames = new String[]{
                "data/balance/balance-scale.train",
                "data/nursery/nursery.data.train",
                "data/led/led.train",
                "data/poker/poker.train"
        };
        String[] testFileNames = new String[]{
                "data/balance/balance-scale.test",
                "data/nursery/nursery.data.test",
                "data/led/led.test",
                "data/poker/poker.test"
        };

        for(int i = 0; i < 4; i++){
            System.out.println("===== Dataset <" + datasetNames[i] + "> =====");

            /* Decision Tree */
            DecisionTree decisionTree = new DecisionTree(trainFileNames[i], testFileNames[i]);
            decisionTree.readDataFromFiles();

            decisionTree.train();
            //decisionTree.printRFStructure();

            decisionTree.evaluateQuality();
            final int[][] DTmatrix = decisionTree.getConfusionMatrix();

            System.out.println("=== Decision Tree's Performance ===");
            matrixToPerformance(DTmatrix);
            System.out.println();

            /* Random Forest */
            RandomForest randomForest = new RandomForest(trainFileNames[i], testFileNames[i], 100);
            randomForest.readDataFromFiles();

            randomForest.train();
            //randomForest.printRFStructure();

            randomForest.evaluateQuality();
            final int[][] RFmatrix = randomForest.getConfusionMatrix();

            System.out.println("=== Random Forest's Performance ===");
            matrixToPerformance(RFmatrix);

            System.out.println("========================================");
            System.out.println();
            System.out.println();
        }
    }

    public static void matrixToPerformance(final int[][] confusionMatrix){
        int n = confusionMatrix.length;

        // Calculate the sum
        int sum = 0;
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                sum += confusionMatrix[i][j];
            }
        }

        /**
         *  For overall performance, output:
         *  Accuracy
         */
        int correctSum = 0;
        for(int i = 0; i < n; i++){
            correctSum += confusionMatrix[i][i];
        }
        double accuracy = correctSum / (double) sum;
        System.out.println("----------------------------------------");
        System.out.println("Overall accuracy:\t" + accuracy);
        System.out.println("----------------------------------------");

        /**
         * For each class, output:
         * Sensitivity, Specificity, Precision, Recall,
         * F-1 Score, F \beta score (\beta = 0.5 and 2)
         */

        double[] sensitivity = new double[n];
        double[] specificity = new double[n];
        double[] precision = new double[n];
        double[] recall = new double[n];
        double[] F1 = new double[n];
        double[] FPoint5 = new double[n];
        double[] F2 = new double[n];

        for(int i = 0; i < n; i++){
            /* Construct a table of confusion */
            int[][] table = new int[][]{{0, 0}, {0, 0}};


            // Calculate TP
            table[0][0] = confusionMatrix[i][i];

            // Calculate P
            int P = 0;
            for(int j = 0; j < n; j++){
                P += confusionMatrix[i][j];
            }

            // Calculate FN
            table[0][1] = P - table[0][0];

            // Calculate P'
            int Pp = 0;
            for(int j = 0; j < n; j++){
                Pp += confusionMatrix[j][i];
            }

            // Calculate FP
            table[1][0] = Pp - table[0][0];

            // Calculate TN
            table[1][1] = sum - table[0][0] - table[0][1] - table[1][0];

            // Calculate N
            int N = table[1][1] + table[1][0];

            /* Sensitivity */
            sensitivity[i] = table[0][0] / (double) P;

            /* Specificity */
            specificity[i] = table[1][1] / (double) N;

            /* Precision */
            precision[i] = table[0][0] / (double) Pp;

            /* Recall */
            recall[i] = sensitivity[i];

            /* F-1 Score */
            F1[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i]);

            /* F \beta score (\beta = 0.5) */
            FPoint5[i] = (1.25 * precision[i] * recall[i]) / (0.25 * precision[i] + recall[i]);

            /* F \beta score (\beta = 2) */
            F2[i] = (5 * precision[i] * recall[i]) / (4 * precision[i] + recall[i]);
        }


        /* Output performance */
        // Format
        DecimalFormat df = new DecimalFormat("0.0000");
        final String NaNString = "NaN     ";

        // Each class
        System.out.print("Class No. \t\t\t");
        for(int i = 0; i < n; i++){
            System.out.print(i + "       ");
        }
        System.out.println();
        System.out.println("----------------------------------------");

        // Sensitivity
        System.out.print("Sensitivity:\t\t");
        for(int i = 0; i < n; i++){
            System.out.print(Double.isNaN(sensitivity[i])? NaNString : df.format(sensitivity[i]) + "\t");
        }
        System.out.println();

        // Specificity
        System.out.print("Specificity:\t\t");
        for(int i = 0; i < n; i++){
            System.out.print(Double.isNaN(specificity[i])? NaNString : df.format(specificity[i]) + "\t");
        }
        System.out.println();

        // Precision
        System.out.print("Precision:\t\t\t");
        for(int i = 0; i < n; i++){
            System.out.print(Double.isNaN(precision[i])? NaNString : df.format(precision[i]) + "\t");
        }
        System.out.println();

        // Recall
        System.out.print("Recall:\t\t\t\t");
        for(int i = 0; i < n; i++){
            System.out.print(Double.isNaN(recall[i])? NaNString : df.format(recall[i]) + "\t");
        }
        System.out.println();

        // F-1 Score
        System.out.print("F-1 Score:\t\t\t");
        for(int i = 0; i < n; i++){
            System.out.print(Double.isNaN(F1[i])? NaNString : df.format(F1[i]) + "\t");
        }
        System.out.println();

        // F-0.5
        System.out.print("F-0.5 Score:\t\t");
        for(int i = 0; i < n; i++){
            System.out.print(Double.isNaN(FPoint5[i])? NaNString : df.format(FPoint5[i]) + "\t");
        }
        System.out.println();

        // F-2
        System.out.print("F-2 Score:\t\t\t");
        for(int i = 0; i < n; i++){
            System.out.print(Double.isNaN(F2[i])? NaNString : df.format(F2[i]) + "\t");
        }
        System.out.println();

        System.out.print("----------------------------------------");
        System.out.println();

    }
}
