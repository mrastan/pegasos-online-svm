package org.ethz.las;

import java.io.*;
import java.util.*;

public class Performance {

  /**
   * Reads the SVM weights obtained from the MapReduce solution
   * and instantiates the Support Vector Machines. Given a file with
   * k lines, it will try to instantiate the average model of k SVMs,
   * each of which is described by one line.
   */
  public static SVM parse(File file) throws IOException {
    List<SVM> svms = new LinkedList<SVM>();
    BufferedReader br = new BufferedReader(new FileReader(file));
    String line = null;
    while ((line = br.readLine()) != null) svms.add(new SVM(line));
    br.close();
    return new SVM(svms);
  }

  /**
   * Given a SVM model and a file with test examples it calculates the
   * accuracy, precision, recall and F1 score. WARNING: if your examples
   * file is too large, this may take a while.
   */
  public static void evaluate(SVM svm, File file) throws IOException {
    int cnt = 1, TP = 0, TN = 0, FP = 0, FN = 0;

    BufferedReader br = new BufferedReader(new FileReader(file));
    String line = null;

    while ((line = br.readLine()) != null) {
      TrainingInstance t = new TrainingInstance(line);
      int trueLabel = t.getLabel();
      int predictedLabel = svm.classify(t);
      if      ((trueLabel == -1) && (predictedLabel == -1)) TN += 1;
      else if ((trueLabel == -1) && (predictedLabel ==  1)) FP += 1;
      else if ((trueLabel == 1)  && (predictedLabel == -1)) FN += 1;
      else if ((trueLabel == 1)  && (predictedLabel ==  1)) TP += 1;

      if (cnt % 1000 == 0) printStats(TP, FN, FP, TN);
      cnt ++;
    }
  }

  /**
   * Prints the learning algorithm statistics.
   */
  public static void printStats(int TP, int FN, int FP, int TN) {
    int cost = 4 * FP + 1 * FN;
    double accuracy = 1.0 * (TP + TN)/(TP + TN + FP + FN);
    double precision = 1.0 * TP/(TP + FP);
    double recall = 1.0 * TP/(TP + FN);
    double f1 = 2 * precision * recall / (precision + recall);
    System.out.println(String.format("Cost = %d CHF, Accuracy: %.3f, Precision = %.3f, recall = %.3f, F1 = %.3f (TP = %d, TN = %d, FP = %d, FN = %d", cost, accuracy, precision, recall, f1, TP, TN, FP, FN));
  }

  /**
   * Make sure to pass the SVM as the first argument and the file containing
   * test instances as the second file.
   */
  public static void main(String[] args) throws Exception {
    SVM svm = parse(new File(args[0]));
    evaluate(svm, new File(args[1]));
  }
}
