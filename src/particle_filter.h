/**
 * particle_filter.h
 * 2D particle filter class.
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include <eigen3/Eigen/Dense>
#include <string>
#include <vector>

#include "helper_functions.h"

struct Particle {
  int id;
  double x;
  double y;
  double theta;
  double weight;
  std::vector<int> associations;
  std::vector<double> sense_x;
  std::vector<double> sense_y;

  /**
   * @brief      Update predicted landmark measurements
   *
   * @param[in]  map               The map
   * @param[in]  sensorRange       The sensor range
   * @param[in]  sensorStd         The sensor standard
   * @param      validLandmarkIds  The indices of valid landmarks that are
   * within the sensor range
   */
  std::pair<Eigen::Matrix2Xd, std::vector<int>> computePredictedObservations(
      const Map &map, double sensorRange, const Eigen::Vector2d &sensorStd);

  std::vector<int> observationAssociation(
      const Eigen::Matrix2Xd &observationsInWorld,
      const Eigen::Matrix2Xd &predictedObservationsInWorld);

  void updateWeight(const Eigen::Matrix2Xd &observationsInWorld,
                    const Eigen::Matrix2Xd &predictedLandmarks,
                    const std::vector<int> &matchedIndices,
                    const Eigen::Vector2d &sensorStd);

  Eigen::Matrix2Xd computeObsvervationsInWorld(
      const std::vector<LandmarkObs> &observations);
};

class ParticleFilter {
 public:
  // Constructor
  // @param num_particles Number of particles
  explicit ParticleFilter(int num_particles = 100)
    : num_particles(num_particles)
    , is_initialized(false) {}

  // Destructor
  ~ParticleFilter() {}

  /**
   * init Initializes particle filter by initializing particles to Gaussian
   *   distribution around first position and all the weights to 1.
   * @param x Initial x position [m] (simulated estimate from GPS)
   * @param y Initial y position [m]
   * @param theta Initial orientation [rad]
   * @param std[] Array of dimension 3 [standard deviation of x [m],
   *   standard deviation of y [m], standard deviation of yaw [rad]]
   * @param num_particles number of particles to be used
   */
  void init(double x, double y, double theta, const Eigen::Vector3d &std);

  /**
   * prediction Predicts the state for the next time step
   *   using the process model.
   * @param delta_t Time between time step t and t+1 in measurements [s]
   * @param std_pose[] Array of dimension 3 [standard deviation of x [m],
   *   standard deviation of y [m], standard deviation of yaw [rad]]
   * @param velocity Velocity of car from t to t+1 [m/s]
   * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
   */
  void prediction(double delta_t, const Eigen::Vector3d &std_pose,
                  double velocity, double yaw_rate);

  /**
   * updateWeights Updates the weights for each particle based on the likelihood
   *   of the observed measurements.
   * @param sensor_range Range [m] of sensor
   * @param std_landmark[] Array of dimension 2
   *   [Landmark measurement uncertainty [x [m], y [m]]]
   * @param observations Vector of landmark observations
   * @param map Map class containing map landmarks
   */
  void updateWeights(double sensor_range, const Eigen::Vector2d &std_landmark,
                     const std::vector<LandmarkObs> &observations,
                     const Map &map_landmarks);

  /**
   * resample Resamples from the updated set of particles to form
   *   the new set of particles.
   */
  void resample();

  /**
   * Set a particles list of associations, along with the associations'
   *   calculated world x,y coordinates
   * This can be a very useful debugging tool to make sure transformations
   *   are correct and assocations correctly connected
   */
  void setAssociations(Particle &particle, const std::vector<int> &associations,
                       const std::vector<double> &sense_x,
                       const std::vector<double> &sense_y);

  /**
   * initialized Returns whether particle filter is initialized yet or not.
   */
  bool initialized() const { return is_initialized; }

  /**
   * Used for obtaining debugging information related to particles.
   */
  std::string getAssociations(Particle best);
  std::string getSenseCoord(Particle best, std::string coord);

  // Set of current particles
  std::vector<Particle> particles;

 private:
  // Number of particles to draw
  const int num_particles;

  // Flag, if filter is initialized
  bool is_initialized;

  // Vector of weights of all particles
  std::vector<double> weights;
};

#endif  // PARTICLE_FILTER_H_