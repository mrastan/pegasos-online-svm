package org.ethz.las;

/**
 * Implements the simple vector logic where each
 * component is a real number.
 */
class RealVector{

  double[] w;

  public RealVector(double[] w) {
    this.w = w;
  }

  public RealVector(int dim) {
    // Will be zeroed by default.
    this.w = new double[dim];
  }

  /**
   * Dot-product between two vectors. Assumes equal dimensions.
   */
  public RealVector multi(RealVector other) {
    double result = 0.0;
    double[] u = other.getFeatures();
    for (int i = 0; i < u.length; ++i)
      this.w[i] *= u[i];
    return this;
  }

  /**
   * Adds a vector to this vector. Assumes equal dimensions.
   */
  public void add(RealVector other) {
    double[] u = other.getFeatures();
    for (int i = 0; i < u.length; ++i)
      this.w[i] += u[i];
  }

  /**
   * Dot-product between two vectors. Assumes equal dimensions.
   */
  public double dotProduct(RealVector other) {
    double result = 0.0;
    double[] u = other.getFeatures();
    for (int i = 0; i < u.length; ++i)
      result += u[i] * this.w[i];
    return result;
  }

  /**
   * Scales the coefficients of this vector by some real factor.
   */
  public RealVector scaleThis(double factor) {
    for (int i = 0; i < this.w.length; ++i)
      this.w[i] *= factor;
    return this;
  }

  /**
   * Creates a new vector which is a scaled version of this vector.
   */
  public RealVector scale(double factor) {
    double[] other = new double[this.w.length];
    for (int i = 0; i < this.w.length; ++i)
      other[i] = this.w[i] * factor;
    return new RealVector(other);
  }

  /**
   * L2 norm of the vector.
   */
  public double getNorm() {
    return Math.sqrt(dotProduct(this));
  }

  public double[] getFeatures() {
    return this.w;
  }

  public int getDimension() {
    return this.w.length;
  }

  public String toString() {
    StringBuffer sb = new StringBuffer();
    for (int i = 0; i < this.w.length; ++i)
      sb.append(w[i] + " ");
    return sb.toString();
  }
}
