/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <algorithm>
#include <array>
#include <boost/format.hpp>
#include <boost/log/trivial.hpp>
#include <cmath>
#include <ctime>
#include <eigen3/Eigen/Geometry>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using Eigen::Isometry2d;
using Eigen::Matrix2Xd;
using Eigen::Vector2d;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::string;
using std::vector;

namespace {

/**
 * @brief      Trivial implementation of nearest neighborhood using euclidean
 * distances
 *
 * @param[in]  obs        The obs
 * @param      landmarks  The landmarks
 *
 * @return     the min distance indices in landmark set
 */
int findNN(const Vector2d &obs, Matrix2Xd &predicted) {
  int minIndex = -1;

  if (predicted.cols() < 1) {
    return minIndex;
  }

  VectorXd distances = (predicted.colwise() - obs).colwise().norm();
  double minDist = distances.minCoeff(&minIndex);
  if (not std::isinf(minDist)) {
    predicted.col(minIndex) = Vector2d(std::numeric_limits<double>::infinity(),
                                       std::numeric_limits<double>::infinity());
    return minIndex;
  } else {
    return -1;
  }
}

double Gaussian(const Vector2d &obs, const Vector2d &landmark,
                const Vector2d &std) {
  using std::exp;
  double dx = obs[0] - landmark[0];
  double dy = obs[1] - landmark[1];
  double gaussianNormalizer = 1.0 / (2 * M_PI * std[0] * std[1]);
  return gaussianNormalizer *
         exp(-(dx * dx / (2 * std[0] * std[0]) + (dy * dy)) /
             (2 * std[1] * std[1]));
}

}  // end of namespace

std::pair<Eigen::Matrix2Xd, std::vector<int>>
Particle::computePredictedObservations(const Map &map, double sensorRange,
                                       const Eigen::Vector2d &sensorStd) {
  std::default_random_engine gen;
  gen.seed(std::time(0));
  std::normal_distribution<double> x_d{0.0, sensorStd[0]};
  std::normal_distribution<double> y_d{0.0, sensorStd[1]};

  vector<int> indices;
  const Vector2d pos = {x, y};

  std::vector<Vector2d, Eigen::aligned_allocator<Vector2d>> landmarks;
  for (size_t i = 0; i < map.landmark_list.size(); ++i) {
    Vector2d landmark = {map.landmark_list[i].x_f + x_d(gen),
                         map.landmark_list[i].y_f + y_d(gen)};
    if ((pos - landmark).norm() < sensorRange) {
      indices.push_back(i);
      landmarks.push_back(landmark);
    }
  }
  return std::make_pair(Eigen::Map<const Matrix2Xd>(
                            reinterpret_cast<const double *>(landmarks.data()),
                            2, landmarks.size()),
                        indices);
}

std::vector<int> Particle::observationAssociation(
    const Matrix2Xd &observationsInWorld,
    const Matrix2Xd &predictedObservationsInWorld) {
  associations.clear();

  // Make a local copy
  vector<int> matches;
  Matrix2Xd predictedObservationsInWorld2 = predictedObservationsInWorld;
  for (int i = 0; i < observationsInWorld.cols(); ++i) {
    int nearestPredicted =
        findNN(observationsInWorld.col(i), predictedObservationsInWorld2);
    if (nearestPredicted != -1) {
      matches.push_back(nearestPredicted);
    }
  }
  return matches;
}

void Particle::updateWeight(const Eigen::Matrix2Xd &observationsInWorld,
                            const Eigen::Matrix2Xd &predictedLandmarks,
                            const std::vector<int> &matchedIndices,
                            const Eigen::Vector2d &sensorStd) {
  BOOST_LOG_TRIVIAL(debug)
      << (boost::format(" --- Updating weight of particle %d --- ") % id).str();

  // Reset weight
  weight = 1.0;

  for (size_t i = 0; i < matchedIndices.size(); ++i) {
    Vector2d observation = observationsInWorld.col(i);
    Vector2d predicted = predictedLandmarks.col(matchedIndices[i]);

    const double likelihood = Gaussian(observation, predicted, sensorStd);

    BOOST_LOG_TRIVIAL(debug)
        << (boost::format("observation=%s, landmark=%s, likelihood=%.3e") %
            observation.transpose() % predicted.transpose() % likelihood)
               .str();
    weight *= likelihood;
  }

  BOOST_LOG_TRIVIAL(debug)
      << (boost::format("Updating weight to %.3e") % weight).str();
}

void ParticleFilter::init(double x, double y, double theta,
                          const Eigen::Vector3d &std) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  BOOST_LOG_TRIVIAL(info) << "Initializing particle filters.";
  std::default_random_engine gen;
  gen.seed(std::time(0));

  std::normal_distribution<double> x_d{x, std[0]};
  std::normal_distribution<double> y_d{y, std[1]};
  std::normal_distribution<double> s_d{theta, std[2]};

  particles = vector<Particle>(num_particles);
  weights = vector<double>(num_particles);
  for (int i = 0; i < num_particles; ++i) {
    Particle &p = particles[i];

    p.id = i;
    p.x = x_d(gen);
    p.y = y_d(gen);
    p.theta = s_d(gen);
    p.weight = 1.0;

    weights[i] = 1.0;
  }

  // Set to true after initialization
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, const Eigen::Vector3d &std_pose,
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  using std::abs;
  using std::cos;
  using std::sin;

  // BOOST_LOG_TRIVIAL(info) << "Run prediction.";
  std::default_random_engine gen;
  gen.seed(std::time(0));

  std::normal_distribution<double> x_d{0.0, std_pose[0]};
  std::normal_distribution<double> y_d{0.0, std_pose[1]};
  std::normal_distribution<double> s_d{0.0, std_pose[2]};

  for (size_t i = 0; i < particles.size(); ++i) {
    Particle &p = particles[i];
    double delta_yaw = 0.0;
    double delta_x = 0.0;
    double delta_y = 0.0;
    if (abs(yaw_rate) < 1e-6) {
      delta_x = velocity * delta_t * cos(p.theta);
      delta_y = velocity * delta_t * sin(p.theta);
    } else {
      delta_x = velocity / yaw_rate * (sin(p.theta + delta_yaw) - sin(p.theta));
      delta_y = velocity / yaw_rate * (cos(p.theta) - cos(p.theta + delta_yaw));
      delta_yaw = yaw_rate * delta_t;
    }
    p.x += delta_x + x_d(gen);
    p.y += delta_y + y_d(gen);
    p.theta += delta_yaw + s_d(gen);
  }
}

Eigen::Matrix2Xd Particle::computeObsvervationsInWorld(
    const std::vector<LandmarkObs> &observations) {
  // Construct eigen matrices for convenient computation
  vector<double> obsBuffer(observations.size() * 2);
  for (size_t i = 0, j = 0; i < observations.size(); ++i, j += 2) {
    obsBuffer[j] = observations[i].x;
    obsBuffer[j + 1] = observations[i].y;
  }

  Matrix2Xd obs =
      Eigen::Map<Matrix2Xd>(obsBuffer.data(), 2, observations.size());
  Isometry2d Tparticle = getIsometry2d(x, y, theta);
  return (Tparticle * obs.colwise().homogeneous()).topRows<2>();
}

void ParticleFilter::updateWeights(double sensor_range,
                                   const Eigen::Vector2d &std_landmark,
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no
   * scaling). The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  // BOOST_LOG_TRIVIAL(debug) << "Updating particle weights.";
  using Eigen::Isometry2d;
  using Eigen::Vector2f;

  // Update sense_x, sense_y and associations.
  for (size_t i = 0; i < particles.size(); ++i) {
    Particle &p = particles[i];

    const Eigen::Matrix2Xd observationsInWorld =
        p.computeObsvervationsInWorld(observations);

    Matrix2Xd predictedObservations;
    vector<int> predictedLandmarkIndices;
    std::tie(predictedObservations, predictedLandmarkIndices) =
        p.computePredictedObservations(map_landmarks, sensor_range,
                                       std_landmark);

    std::vector<int> matchedIndices =
        p.observationAssociation(observationsInWorld, predictedObservations);

    p.associations.clear();
    for (int matchedIndex : matchedIndices) {
      // +1 to output the landmark id
      p.associations.push_back(predictedLandmarkIndices[matchedIndex] + 1);
    }

    p.updateWeight(observationsInWorld, predictedObservations, matchedIndices,
                   std_landmark);

    weights[i] = p.weight;
  }
}

void ParticleFilter::resample() {
  using namespace std;
  if (weights.empty()) {
    return;
  }

  VectorXd weights_ =
      Eigen::Map<VectorXd>(weights.data(), weights.size(), 1).normalized();
  double maxWeight = weights_.maxCoeff();

  cout << weights_.transpose() << endl;

  // First index initialization
  std::default_random_engine gen;
  gen.seed(std::time(0));
  std::uniform_real_distribution<> urd(0, 1.0);
  std::uniform_int_distribution<> rid(0, num_particles - 1);

  int sampledIndex = rid(gen);

  // Resampling the particles
  vector<Particle> newParticles(num_particles);

  double beta = 0.0;
  for (int i = 0; i < weights_.size(); ++i) {
    beta += urd(gen) * 2.0 * maxWeight;
    while ((beta - weights_[sampledIndex]) > 1e-6) {
      beta -= weights_[sampledIndex];
      sampledIndex = (sampledIndex + 1) % weights_.rows();
    }
    newParticles[i] = particles[sampledIndex];
    newParticles[i].id = i;
  }

  particles = newParticles;
}

void ParticleFilter::setAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}