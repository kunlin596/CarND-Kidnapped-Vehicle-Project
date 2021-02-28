/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <array>
#include <eigen3/Eigen/Geometry>
#include <boost/log/trivial.hpp>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::cout;
using std::endl;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  BOOST_LOG_TRIVIAL(info) << "Initializing particle filters.";
  std::random_device rd {};
  std::mt19937 gen { rd() };
  std::normal_distribution<double> x_d { x, std[0] };
  std::normal_distribution<double> y_d { y, std[1] };
  std::normal_distribution<double> s_d { theta, std[2] };

  num_particles = 5;  // TODO: Set the number of particles
  particles = std::vector<Particle>(num_particles);
  for (int i = 0; i < num_particles; i++) {
    Particle &p = particles[i];
    p.id = i;
    p.x = x_d(gen);
    p.y = y_d(gen);
    p.theta = s_d(gen);
    p.weight = 1.0 / static_cast<double>(num_particles);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  using std::sin;
  using std::cos;

  BOOST_LOG_TRIVIAL(info) << "Run prediction.";
  static std::random_device rd {};
  static std::mt19937 gen { rd() };
  static std::normal_distribution<double> x_d { 0.0, std_pos[0] };
  static std::normal_distribution<double> y_d { 0.0, std_pos[1] };
  static std::normal_distribution<double> s_d { 0.0, std_pos[2] };

  for (size_t i = 0; i < particles.size(); ++i) {
    Particle &p = particles[i];
    if (0.0 < yaw_rate and yaw_rate < 1e-6) {
      yaw_rate = 1e-6;
    } else if (-1e-6 < yaw_rate and yaw_rate < 0.0) {
      yaw_rate = -1e-6;
    }
    const double delta_yaw = yaw_rate * delta_t;
    const double delta_x = velocity / yaw_rate * (sin(p.theta + delta_yaw) - sin(p.theta));
    const double delta_y = velocity / yaw_rate * (cos(p.theta) - cos(p.theta + delta_yaw));
    p.x += delta_x + x_d(gen);
    p.y += delta_y + y_d(gen);
    p.theta += delta_yaw + s_d(gen);
  }
}

void ParticleFilter::dataAssociation(const vector<LandmarkObs> &predictedInWorld,
                                     const vector<LandmarkObs> &observations) {
  /**
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   *
   *   Data association is done in world frame.
   *   Observations are transformed from car frame to world frame
   */

  // BOOST_LOG_TRIVIAL(debug) << "Data association";
  using Eigen::Vector2d;
  using Eigen::Isometry2d;

  for (auto &p : particles) {
    p.associations.clear();
    vector<LandmarkObs> predictedPerParticle(predictedInWorld.size());
    std::copy(predictedInWorld.begin(), predictedInWorld.end(),
              std::back_inserter(predictedPerParticle));
    Isometry2d particleTransform = getIsometry2d(p.x, p.y, p.theta);
    for (auto &o : observations) {
      // Find out the NN predictions
      int closest = -1;
      double minDist = std::numeric_limits<double>::infinity();
      for (size_t i = 0; i < predictedPerParticle.size(); ++i) {
        Vector2d obsInWorld = (particleTransform * Vector2d(o.x, o.y).homogeneous()).topRows<2>();
        const double dist = (Vector2d(predictedPerParticle[i].x, predictedPerParticle[i].y) - obsInWorld).norm();
        if (dist < minDist) {
          minDist = dist;
          closest = i;
        }
      }

      // Cache the index of NN and remove it from predictions
      if (closest != -1) {
        p.associations.push_back(closest);
        predictedPerParticle.erase(predictedPerParticle.begin() + closest);
      }
    }
  }
}

static double computeLikelyhood(Eigen::Vector2d x, Eigen::Vector2d u, Eigen::Vector2d std) {
  return 1.0 / (2 * M_PI * std[0] * std[1]) * std::exp(
      -(
          std::pow((x[0] - u[0]), 2) / (2 * std[0] * std[0]) +
          std::pow((x[1] - u[1]), 2) / (2 * std[1] * std[1])
      )
  );
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  BOOST_LOG_TRIVIAL(debug) << "Updating particle weights.";
  using Eigen::Vector2d;
  using Eigen::Vector2f;
  using Eigen::Isometry2d;

  // Update data associations
  vector<LandmarkObs> predictedObservations(map_landmarks.landmark_list.size());
  for (size_t i = 0; i < map_landmarks.landmark_list.size(); ++i) {
    LandmarkObs &obs = predictedObservations[i];
    obs.x = static_cast<double>(map_landmarks.landmark_list[i].x_f);
    obs.y = static_cast<double>(map_landmarks.landmark_list[i].y_f);
  }

  dataAssociation(predictedObservations, observations);

  // Update weights
  const Vector2d stddev = {std_landmark[0], std_landmark[1]};
  for (auto &p : particles) {
    // const Isometry2d Tcar2world = getIsometry2d(p.x, p.y, p.theta);
    double &prob = p.weight;
    for (size_t i = 0; i < observations.size(); ++i) {
      const LandmarkObs &o = observations[i];
      const Map::single_landmark_s &nnLandmark = map_landmarks.landmark_list[p.associations[i]];
      const Vector2d observation = { o.x, o.y };
      const Vector2d predictedObservation = Vector2f(nnLandmark.x_f, nnLandmark.y_f) .cast<double>() - Vector2d(p.x, p.y);
      prob *= computeLikelyhood(observation, predictedObservation, stddev);
    }
  }
}

void ParticleFilter::resample() {
  // Clear the old weights
  weights = vector<double>(num_particles);

  // Find out the max weight for resampling step
  double maxweight = std::numeric_limits<double>::min();
  for (size_t i = 0; i < num_particles; ++i) {
    double &w = particles[i].weight;
    if (w > maxweight) {
      maxweight = w;
    }
  }

  // First index initialization
  std::random_device rd {};
  std::mt19937 gen { rd() };
  std::uniform_real_distribution<double> urd(0, 1.0);
  std::uniform_int_distribution<int> rid(0, num_particles - 1);
  int currindex = rid(gen);

  // Resampling the particles
  int newNumParticles= 0;
  vector<Particle> newParticles(particles.size());
  while (newNumParticles < num_particles) {
    double beta = urd(gen) * 2.0 * maxweight;
    while ((beta - particles[currindex].weight) > 1e-6) {
      beta -= particles[currindex].weight;
      currindex = (currindex + 1) % num_particles;
    }
    newParticles[newNumParticles] = particles[currindex];
    ++newNumParticles;
  }
  particles = newParticles;
}

void ParticleFilter::setAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
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
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}