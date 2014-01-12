package org.ethz.las;

import java.util.*;

public class SVM {

  // Hyperplane weights.
  RealVector weights;

  public SVM(RealVector weights) {
    this.weights = weights;
  }

  /**
   * Instantiates an SVM from a list of training instances, for a given
   * learning rate 'eta' and regularization parameter 'lambda'.
   */
  public SVM(List<TrainingInstance> trainingSet, double lambda, double kf, int nepochs) {
    int dim = trainingSet.get(0).getFeatureCount();
    int k = (int)(trainingSet.size() * kf);

    weights = new RealVector(dim);

    for (int n=1; n <= nepochs; n++) {
      // learning rate 
      double eta_n = 1.0 / (lambda*(n+2));

      RealVector w_n = new RealVector(weights.getFeatures());
      w_n.scaleThis(1.0 -(eta_n*lambda));

      //Random R = new Random();

      // calc sub-gradients
      //for (int j = 0; j < k; j++) {
      for (TrainingInstance ti : trainingSet) {
        // choose random example
        //TrainingInstance ti = trainingSet.get( R.nextInt(trainingSet.size() - 1) );

        // calculate prediction
        double score=weights.dotProduct( ti.getFeatures() );        

        if (score * ti.getLabel() < 1) {
          if (ti.getLabel() < 0) {
            w_n.add( ti.getFeatures().scale((double)ti.getLabel() * eta_n) );  
          } else {
            w_n.add( ti.getFeatures().scale((double)ti.getLabel() * eta_n) );  
          }
        }
      }
      
      double norm2 = w_n.getNorm();
      w_n.scaleThis(Math.min(1.0,(1/Math.sqrt(lambda))/norm2));
      weights = new RealVector(w_n.getFeatures());
    }

  }

  /**
   * Instantiates SVM from weights given as a string.
   */
  public SVM(String w) {
    List<Double> ll = new LinkedList<Double>();
    Scanner sc = new Scanner(w);
    while(sc.hasNext()) {
      double coef = sc.nextDouble();
      ll.add(coef);
    }

    double[] weights = new double[ll.size()];
    int cnt = 0;
    for (Double coef : ll)
      weights[cnt++] = coef;

    this.weights = new RealVector(weights);
  }

  /**
   * Instantiates the SVM model as the average model of the input SVMs.
   */
  public SVM(List<SVM> svmList) {
    int dim = svmList.get(0).getWeights().getDimension();
    RealVector weights = new RealVector(dim);
    for (SVM svm : svmList)
      weights.add(svm.getWeights());

    this.weights = weights.scaleThis(1.0/svmList.size());
  }

  /**
   * Given a training instance it returns the result of sign(weights'instanceFeatures).
   */
  public int classify(TrainingInstance ti) {
    RealVector features = ti.getFeatures();
    double result = ti.getFeatures().dotProduct(this.weights);
    if (result >= 0) return 1;
    else return -1;
  }

  public RealVector getWeights() {
    return this.weights;
  }

  @Override
  public String toString() {
    return weights.toString();
  }
}
