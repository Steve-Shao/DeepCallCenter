#ifndef TEST_H
#define TEST_H

#include <set>
#include <iostream>
#include <math.h>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>
#include <numeric>
#include <limits>
#include <list>
#include <vector>
#include <deque>
#include <stdlib.h>
#include <stdio.h>
#include <array>
#include <fcntl.h>
#include <sys/stat.h>
#include <stdio.h>
#include <unistd.h>
#include <utility>
#include <cstdio>
#include <sstream>
#include <map>
#include <chrono>
#include <omp.h>
#include <random>

namespace simulation
{
	class Simulation
	{
		public: 
			Simulation();
			~Simulation();

			int save();

			std::vector<std::string> splitString(const std::string& input, char delimiter);
			std::vector<std::vector<double>> readMatrixFromCSV(const std::string& filename); 
			std::vector<double> readVectorFromCSV(const std::string& filename);

		private:

			int class_no;
			int num_interval;
			int decision_freq;
			std::vector<int> upper;

			std::vector<double> lambda;
			std::vector<int> no_server; 

			std::vector<double> mu_hourly;
			std::vector<double> theta_hourly;

			std::vector<double> holding_cost_rate;
			std::vector<double> abandonment_cost_rate;
			std::vector<double> cost_rate;

			std::vector<std::vector<double>> arr_cdf; 
			std::vector<std::vector<double>> lambda_trans;
	};


	class MyNetwork
	{
	  
	  public:
		std::vector<float> gamma;
		std::vector<float> beta;
		std::vector<float> mu;
		std::vector<float> batchnorm_denom;
	    std::vector<std::vector<float> > w1;
		std::vector<float> b1;
		std::vector<std::vector<float> > w2;
		std::vector<float> b2;
		std::vector<std::vector<float> > w3;
		std::vector<float> b3;
		std::vector<std::vector<float> > w4;
		std::vector<float> b4;
		std::vector<std::vector<float> > w5;
		std::vector<float> b5;
		std::vector<std::vector<float> > w6;
		std::vector<std::vector<float> > forward(const std::vector<std::vector<float> >& x);
		void load(const std::string &filename);
		std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>> &matrix);
		
		~MyNetwork();
		
		private:
	
			std::vector<std::vector<float> > add(const std::vector<std::vector<float> >& x, std::vector<float>& y);
			std::vector<std::vector<float> > matmul(const std::vector<std::vector<float> >& mat1, const std::vector<std::vector<float> >& mat2);
			std::vector<std::vector<float> > divide(const std::vector<std::vector<float> >& x, std::vector<float>& y);
			std::vector<std::vector<float> > subtract(const std::vector<std::vector<float> >& x, std::vector<float>& y);
			std::vector<std::vector<float> > multiply(const std::vector<std::vector<float> >& x, std::vector<float>& y);
			std::vector<std::vector<float> > leaky_relu(const std::vector<std::vector<float> >& x);
			std::vector<std::vector<float> > readMatrix(const std::string &filename);
			std::vector<float> readVector(const std::string &filename);

	};


	class Execute
	{
		public: 
			Execute(int& class_no_,
							std::vector<std::vector<double>>& arr_cdf_,
							std::vector<double>& lambda_,
							std::vector<double>& mu_hourly_,
							std::vector<double>& theta_hourly_, 
							std::vector<int>& no_server_,
							std::vector<double>& holding_cost_rate_,
							std::vector<double>& abandonment_cost_rate_,
							std::vector<double>& cost_rate_,
							std::vector<std::vector<double>>& lambda_trans_,
							int& num_interval_,
							int& decision_freq_,
							int& i);
			~Execute();

			std::vector<double> run();
			
		private:

			void queue_init();
			double generate_interarrival(int& interval);
			double generate_abandon(int& cls);
			double generate_service();

			void add_queue(double& arr_time, int& cls);
			void remove_queue(int& cls);


			void handle_arrival_event(int& interval, int& cls, int& pre_interval, int& post_interval);
			void handle_depart_event(int& interval, int& cls, int& pre_interval, int& post_interval);
			void handle_abandon_event(int& interval, int& pre_interval, int& post_interval);
			std::vector<double> argsort(const std::vector<double>&);
			std::vector<double> queueing_discipline(std::vector<int>& num_in_system, int& interval);
			std::vector<float> load_and_predict(std::string &filename, std::vector<std::vector<float> >& input_tensor);

			double t_arrival;
			double t_depart;
			double t_abandon;
			double t_event;
			double sim_clock = 0;
            double T = 17;
            double total_cost;

			int class_abandon;
			int cust_abandon;
			int interval;
			int pre_interval;
			int post_interval;
			int num_interval;
			int decision_freq;
			int class_no;
			
			std::vector<int> num_in_system;
			std::vector<int> num_in_service;
			std::vector<int> num_in_queue;
			std::vector<int> num_abandons;
			std::vector<int> num_arrivals;

			std::vector<double> queue_integral;
			std::vector<double> service_integral;
			std::vector<double> system_integral;
			std::vector<double> holding_cost;
			std::vector<double> waiting_cost;

			std::vector<std::deque<double>> queue_list;
			std::vector<std::vector<double>> abandonment_list;

			std::deque<double> empty_queue;
			std::vector<std::deque<double> > arr_list;
			
			double inf = std::numeric_limits<double>::infinity();
			std::mt19937 generator;

			std::vector<double> mu_hourly;
			std::vector<double> theta_hourly;
			std::vector<double> lambda;
			std::vector<int> no_server; 
			std::vector<double> holding_cost_rate;
			std::vector<double> abandonment_cost_rate;
			std::vector<double> cost_rate;
			std::vector<std::vector<double>> arr_cdf; 
			std::vector<std::vector<double>> lambda_trans;

			simulation::MyNetwork* nn_zs = nullptr; // Declare as a pointer
    		int network_size; // Declare the size as an external variable
			
			
	};
};
#endif
