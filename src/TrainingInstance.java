package org.ethz.las;

import java.util.LinkedList;
import java.util.Scanner;
import java.util.List;

/**
 * Represents a training instance.
 */
class TrainingInstance {

  RealVector features;
  int label;

  public TrainingInstance(RealVector features, int label) {
    this.features = features;
    this.label = label;
  }

  /**
   * Instantiates the training instance from a string.
   * Supposes that the instance is given as a series of doubles and
   * that the last element is the label. To avoid precision problems,
   * the label is considered 1 if the last coefficient is > 0.5, -1 otherwise.
   */
  public TrainingInstance(String s) {
    Scanner sc = new Scanner(s);
    List<Double> parsedInput = new LinkedList<Double>();

    // Gets all tokens.
    int i = 0;
    while (sc.hasNextDouble()) {
      if (i != 0 ) parsedInput.add(sc.nextDouble());
      i++;
    }

    // Last element is always the label.
    int n = parsedInput.size() - 1;

    // Convert the tokens to feature vector and label.
    double [] coef = new double[n];
    int cnt = 0;
    for (Double c : parsedInput) {
      if (cnt < n) coef[cnt++] = c;
      else this.label = c > 0.5 ? 1 : -1;
    }

    // Now set the feature vector.
    this.features = new RealVector(coef);
  }

  public RealVector getFeatures() {
    return features;
  }

  public int getLabel() {
    return label;
  }

  public int getFeatureCount() {
    return features.getDimension();
  }
}
