#include "test.h"

namespace simulation {

	Simulation::Simulation(){
		// 1. Problem dimensions and parameters
		class_no = 17; 										// Number of customer classes
		num_interval = 204;									// Number of intervals we consider to estimate the system parameters

		// 2. Load Data from Files
		try{
			// Arrival rates (per 5-minute intervals)
			lambda = readVectorFromCSV("data/main_test_total_arrivals_partial_5min.csv");

			// Number of agents (converted to integers for C++ purposes)
			std::vector<double> agents = readVectorFromCSV("data/main_test_agents.csv");
			no_server.resize(agents.size());
        	std::transform(agents.begin(), agents.end(), no_server.begin(), [](double a) { return static_cast<int>(a); });

			// Service and abandonment rates (hourly)
			mu_hourly = readVectorFromCSV("data/mu_hourly_17dim.csv"); 
			theta_hourly = readVectorFromCSV("data/theta_hourly_17dim.csv"); 

			// Cost-related data
			holding_cost_rate = readVectorFromCSV("data/hourly_holding_cost_17dim.csv"); 
			abandonment_cost_rate = readVectorFromCSV("data/abandonment_cost_17dim.csv"); 
			cost_rate = readVectorFromCSV("data/hourly_total_cost_17dim.csv"); 

			// Cumulative distribution function (arrival probabilities)
			arr_cdf = readMatrixFromCSV("data/cdf_17dim.csv");
		} catch (const std::exception& e) {
			std::cerr << "Error loading input files: " << e.what() << std::endl;
			throw;
    	}
	}

	// Splits a string into a vector of substrings based on the specified delimiter.
	// Example: splitString("a,b,c", ',') returns {"a", "b", "c"}.
	std::vector<std::string> Simulation::splitString(const std::string& input, char delimiter) {
	    std::vector<std::string> tokens;
	    std::string token;
	    std::istringstream tokenStream(input);

		// Extract tokens until the end of the input string
	    while (std::getline(tokenStream, token, delimiter)) {
	        tokens.push_back(token);
		}

		return tokens;
	}

	// Reads a matrix from a CSV file.
	// Each line in the file represents a row of the matrix, with elements separated by commas.
	std::vector<std::vector<double> > Simulation::readMatrixFromCSV(const std::string& filename) {
	    std::vector<std::vector<double> > matrix;

		// Open the file for reading
	    std::ifstream file(filename);
	    if (!file.is_open()) {
	        std::cerr << "Failed to open the file: " << filename << std::endl;
	        return matrix;
	    }

	    std::string line;
	    while (std::getline(file, line)) {
			
			if (line.empty()) {
            	continue;  // Skip empty lines
        	}

			// Split the line into tokens and convert them to doubles
	        std::vector<std::string> row = splitString(line, ',');
	        std::vector<double> matrixRow;
	        
			try {
            	for (const std::string& str : row) {
                	matrixRow.push_back(std::stod(str));  // Convert each token to a double
            	}
        	} catch (const std::invalid_argument& e) {
            	std::cerr << "Invalid data in file: " << filename << " -> " << e.what() << std::endl;
            	continue;  // Skip rows with invalid data
        	} catch (const std::out_of_range& e) {
            	std::cerr << "Out-of-range data in file: " << filename << " -> " << e.what() << std::endl;
            	continue;  // Skip rows with out-of-range data
        	}

			// Add the row to the matrix
	        matrix.push_back(matrixRow);
	    }

	    file.close();
	    return matrix;
	}

	// Reads a CSV file and returns its contents as a 1D vector of doubles.
	// Assumes the file contains numeric values separated by commas.
	std::vector<double> Simulation::readVectorFromCSV(const std::string& filename) {
	    std::vector<double> vec;
	    
		// Open the file
    	std::ifstream file(filename);
    	if (!file.is_open()) {
        	std::cerr << "Failed to open the file: " << filename << std::endl;
        	return vec;
    	}

	    std::string line;
    	while (std::getline(file, line)) {
        	if (line.empty()) {
            	continue; // Skip empty lines
        	}

        	// Split the line into tokens and convert each to a double
        	std::istringstream lineStream(line);
       		std::string cell;
        	try {
            	while (std::getline(lineStream, cell, ',')) {
                	vec.push_back(std::stod(cell)); // Convert string to double
            	}
        	} catch (const std::invalid_argument& e) {
            	std::cerr << "Invalid data in file: " << filename << " -> " << e.what() << std::endl;
       	 	} catch (const std::out_of_range& e) {
            	std::cerr << "Out-of-range data in file: " << filename << " -> " << e.what() << std::endl;
        	}
    	}

	    file.close();
	    return vec;
	}
	
	Simulation::~Simulation(){
		std::cout << "Done" << std::endl;
	}

	int Simulation::save(){
        // Define the file path where results will be saved
		
		std::string file_name_pol = "/home/ekasikar/main_test_problem_benchmarks/static_benchmarks/cost_current_implementation.csv";
		const char *path_pol = &file_name_pol[0];
		const int myfile_pol = open(path_pol, O_CREAT | O_WRONLY);
		
		// Check if the file opened successfully
		if (myfile_pol != -1){			
			// Parallelized loop for running simulations and saving results
			#pragma omp parallel for num_threads(100)
			for (int i = 0; i < 10000; i++){
				std::vector<double> cost; 
				cost.assign(class_no + 1, 0);
				std::cout << "iter " << i << std::endl;
				simulation::Execute exec(class_no, arr_cdf, lambda, mu_hourly, theta_hourly, no_server,
          									holding_cost_rate, abandonment_cost_rate, cost_rate, num_interval, i); 

				cost = exec.run();				// Execute the simulation and get the result

				std::string results;
				results += std::to_string(i);	// Add iteration index
				results += ",";
				for (int i = 0; i < class_no; i++){
					results += std::to_string(cost[i]);
					results += ",";
				}
				results += std::to_string(cost[class_no]); // Total cost
				results += "\n";
				// Convert the results string to a C-style string and write to the file
				const char *char_results = const_cast<char*>(results.c_str());
				write(myfile_pol, char_results, results.length());
			}
		}
		// Close the file after all iterations are complete
		close(myfile_pol);
		return 0;
	}

	Execute::Execute(int& class_no_,
							std::vector<std::vector<double>>& arr_cdf_,
							std::vector<double>& lambda_,
							std::vector<double>& mu_hourly_,
							std::vector<double>& theta_hourly_, 
							std::vector<int>& no_server_,
							std::vector<double>& holding_cost_rate_,
							std::vector<double>& abandonment_cost_rate_,
							std::vector<double>& cost_rate_,
							int& num_interval_,
							int& i)
	{
		class_no = class_no_;
		arr_cdf = arr_cdf_; 
		lambda = lambda_;
		no_server = no_server_; 

		mu_hourly = mu_hourly_; 
		theta_hourly = theta_hourly_;

		holding_cost_rate = holding_cost_rate_;
		abandonment_cost_rate = abandonment_cost_rate_;
		cost_rate = cost_rate_;
		num_interval = num_interval_;
        
      	generator.seed(i);
		queue_init();
	}

	Execute::~Execute(){}

	void Execute::queue_init(){
		// Initialize queues for each class
		queue_list.resize(class_no);
		arr_list.resize(class_no);

		// Initialize abandonment list with a dummy "infinity" value for each class
		std::vector<double> dummy_abandonment = {inf};
    	abandonment_list.resize(class_no, dummy_abandonment);
	}

	double Execute::generate_interarrival(int& interval){	
		// Generate interarrival time using exponential distribution
		double arrival_rate = lambda[interval] * 12; // Retrieve the hourly arrival rate
    	std::exponential_distribution<double> Arrival(arrival_rate);
    	return Arrival(generator);   
	}

	double Execute::generate_abandon(int& cls) {
		// Generate a random abandonment time using an exponential distribution
    	double abandonment_rate = theta_hourly[cls]; // Retrieve the abandonment rate for the class
    	std::exponential_distribution<double> Abandon(abandonment_rate);
    	return Abandon(generator);
	}

	double Execute::generate_service(){
		double service_rate = 0;				
		for (int i = 0; i < class_no; ++i){
			//hourly service rate
			service_rate += num_in_service[i+1] * mu_hourly[i]; //sumproduct of num_in_service and mu_hourly
		}
	    	std::exponential_distribution<double> Tau(service_rate);
	    	return Tau(generator);
	}

	std::vector<double> Execute::queueing_discipline(std::vector<int>& num_in_system, int& interval){

		std::vector<double> mu_theta_diff;
		std::vector<double> c_mu_theta_diff;
		std::vector<double> c_mu_theta;
		std::vector<double> c_mu;

		mu_theta_diff.assign(class_no, 0);
		c_mu_theta_diff.assign(class_no, 0);
		c_mu_theta.assign(class_no, 0);
		c_mu.assign(class_no, 0);

		
		for (int i = 0; i < class_no; i++){
			c_mu_theta[i] = cost_rate[i]*mu_hourly[i]/theta_hourly[i];
			mu_theta_diff[i] = mu_hourly[i] - theta_hourly[i];
			c_mu_theta_diff[i] = cost_rate[i] * (mu_hourly[i] - theta_hourly[i]);
			c_mu[i] = cost_rate[i] * mu_hourly[i];
		}

		std::vector<double> kappa;

		for (int i = 0; i < class_no; i++) {
			//kappa.push_back(-1 * (c_mu_theta[i])); //c*mu/theta
			//kappa.push_back(-1 * (mu_theta_diff[i])); 
			//kappa.push_back(-1 * (c_mu_theta_diff[i])); 
			//kappa.push_back(-1 * (c_mu[i])); 
			kappa.push_back(-1 * cost_rate[i]);
		}
		std::vector<double> priority_order = argsort(kappa);
		return priority_order;
	}

	void Execute::handle_arrival_event(int& interval, int& cls, int& pre_interval, int& post_interval){ 
		// Update system-wide and class-specific state variables for arrivals
		num_in_system[0] += 1; // increase the total number of people in the system
    	num_in_system[cls + 1] += 1; // increase the number of people in the system from class 'cls'
    	num_arrivals[0] += 1; // increase the total number of arrivals into the system
    	num_arrivals[cls + 1] += 1; // increase the number of arrivals to the class 'cls'

		// Schedule the next arrival
		t_arrival = sim_clock + generate_interarrival(interval);

		// First add the arriving customer to the queue to determine whether to accept to service or keep in the queue
		num_in_queue[0] += 1;
    	num_in_queue[cls + 1] += 1;
    	add_queue(t_event, cls); // update the queue based on this person's arrival time to determine their order in the queue

		// Check if service is possible immediately
		if ((num_in_system[0] <= no_server[interval] && pre_interval == post_interval) || ((num_in_service[0] < no_server[interval]) && pre_interval == post_interval)) {
			if (num_in_service[0] < no_server[interval]) {
			    // Admit to service without waiting 
			    num_in_service[0] += 1;
			    num_in_service[cls + 1] += 1;
			    num_in_queue[0] -= 1;
			    num_in_queue[cls + 1] -= 1;
				remove_queue(cls); // Remove from queue
				t_depart = sim_clock + generate_service(); // Schedule next service completion
			} else if (num_in_system[0] == 0){
			    t_depart = std::numeric_limits<double>::infinity(); // No customers in the system
			}
		}

		else {
			// Handle preemptive scheduling based on the optimal policy
			std::vector<double> priority_order;
			int opt_policy;

			// Adjust service and queue states based on priority
			int avail_server_num; 	// number of available agents
			int num_served = 0; 	// number of customers being served
			std::vector<int> optimal_policy(class_no, 0);

			priority_order = queueing_discipline(num_in_system, interval);

			for (int i = 0; i < priority_order.size(); i++){
				int ind = priority_order[i]; 													// Class based on priority order
				avail_server_num = std::max(no_server[interval] - num_served, 0);				// Number of agents available 
				optimal_policy[ind] = std::max(num_in_system[ind + 1] - avail_server_num, 0);	// The optimal number of people from this class in the queue (Number of people in the system from that class - number of available)
				num_served += std::min(num_in_system[ind + 1], avail_server_num);				// The number of customers being served is equal to minimum of number of people in the system and number of available servers
			}

			int* diff = new int[class_no];

			// Calculate the difference between current queue length and the optimal queue length
			for (int i = 0; i < class_no; i++){
				diff[i] = num_in_queue[i+1] - optimal_policy[i]; // difference -- if positive, people should not be placed in the queue and should be send to service; if negative, more people should be placed in the queue.
				num_in_service[i + 1] += diff[i]; // 
				num_in_queue[0] -= diff[i];
				num_in_queue[i + 1] -= diff[i];

				if (diff[i] >= 0){
					// Admit customers to service from class i 
					for (int j = 0; j < diff[i]; j++){
						remove_queue(i); // remove them from their corresponding queue
					}
				} else if (diff[i] < 0){
					// Add customers back to the queue
					for (int j = 0; j < abs(diff[i]); j++){
						add_queue(t_event,i);
					}
				}
			}

     	 	// Update total number of people in service
     	 	num_in_service[0] = 0;
     	 	for (int i = 0; i < class_no; i++){
     	 		num_in_service[0] += num_in_service[i + 1];
     	 	}
			
			// Schedule the next departure
			if (num_in_service[0] == 0) {
    			t_depart = std::numeric_limits<double>::infinity();
			} else {
    			t_depart = sim_clock + generate_service();
			}
			
			delete[] diff;
		}
	}
	
	void Execute::handle_depart_event(int& interval, int& cls, int& pre_interval, int& post_interval){ 
		// Update system-wide and class-specific state variables for departures
		num_in_system[0] -= 1; // Decrease the total number of people in the system
		num_in_system[cls + 1] -=1; // Decrease the number of people in the system from class 'cls'
		num_in_service[0] -= 1; // Decrease the total number of people in service
		num_in_service[cls + 1] -= 1; // Decrease the number of people in service from class 'cls'
		num_departs[0] += 1; // Increase the total number of people departed
        
		// If there are enough servers and the intervals match, serve the next customer in the queue if possible
		if (num_in_system[0] < no_server[interval] && pre_interval == post_interval){
			if (num_in_queue[0] > 0 && num_in_service[0] < no_server[interval]){
				// Admit the next customer into service
				num_in_service[0] += 1; // Increase the total number of people in service 
				num_in_service[cls + 1] += 1; // Increase the total number of people in service from class 'cls'
				num_in_queue[0] -= 1; // Decrease the total number of people in queue
				num_in_queue[cls + 1] -= 1; // Decrease the number of people in queue from class 'cls'
				remove_queue(cls);
			}

			// Schedule the next departure 
			if (num_in_system[0] > 0){
				t_depart = sim_clock + generate_service();
			} else {            
				t_depart = std::numeric_limits<double>::infinity(); // No more departures; next event is an arrival
			}
		}

		else {
			// Handle preemptive scheduling based on the optimal policy
			std::vector<double> priority_order;
			int opt_policy;
			
			priority_order = queueing_discipline(num_in_system, interval);
			
			// Adjust service and queue states based on priority
			int avail_server_num; 	// number of available agents
			int num_served = 0; 	// number of customers being served
			std::vector<int> optimal_policy(class_no, 0);

			for (int i = 0; i < priority_order.size(); i++){
				int ind = priority_order[i]; 													// Class based on priority order
				avail_server_num = std::max(no_server[interval] - num_served, 0);				// Number of agents available 
				optimal_policy[ind] = std::max(num_in_system[ind + 1] - avail_server_num, 0);	// The optimal number of people from this class in the queue (Number of people in the system from that class - number of available)
				num_served += std::min(num_in_system[ind + 1], avail_server_num);				// The number of customers being served is equal to minimum of number of people in the system and number of available servers
			}

			int* diff = new int[class_no];

			// Calculate the difference between current queue length and the optimal queue length
			for (int i = 0; i < class_no; i++){
				diff[i] = num_in_queue[i+1] - optimal_policy[i]; // difference -- if positive, people should not be placed in the queue and should be send to service; if negative, more people should be placed in the queue.
				num_in_service[i + 1] += diff[i]; // 
				num_in_queue[0] -= diff[i];
				num_in_queue[i + 1] -= diff[i];

				if (diff[i] >= 0){
					// Admit customers to service from class i 
					for (int j = 0; j < diff[i]; j++){
						remove_queue(i); // remove them from their corresponding queue
					}
				} else if (diff[i] < 0){
					// Add customers back to the queue
					for (int j = 0; j < abs(diff[i]); j++){
						add_queue(t_event,i);
					}
				}
			}

			// Update total number of people in service
     	 	num_in_service[0] = 0;
     	 	for (int i = 0; i < class_no; i++){
     	 		num_in_service[0] += num_in_service[i + 1];
     	 	}
			
			
			// Schedule the next departure
			if (num_in_service[0] == 0) {
    			t_depart = std::numeric_limits<double>::infinity();
			} else {
    			t_depart = sim_clock + generate_service();
			}
			
			delete[] diff;
		}
	}

	void Execute::handle_abandon_event(int& interval, int& pre_interval, int& post_interval){
		// Remove the abandoned customer from the system
		num_in_system[0] -= 1; 									// Decrease the total number of people in the system
		num_in_system[class_abandon + 1] -=1; 					// Decrease the number of people in the system from the class abandoned
		num_abandons[0] += 1; 									// Increase the total number of people abandoned
		num_abandons[class_abandon + 1] += 1; 					// Increase the number of people abandoned from class 'class_abandon'
		num_in_queue[0] -= 1; 									// Decrease the total number of people in the queue 
		num_in_queue[class_abandon + 1] -= 1;					// Decrease the number of people in the queue from class 'class_abandon'
		
		// Remove the customer from the queue and abandonment list
		queue_list[class_abandon].erase(queue_list[class_abandon].begin() + cust_abandon);
    	abandonment_list[class_abandon].erase(abandonment_list[class_abandon].begin() + cust_abandon + 1);

		std::vector<double> min_temp(class_no, std::numeric_limits<double>::max());
		
		if (num_in_queue[0] > 0){
			//find the minimum abandonment time from all the queues 
			for (int i = 0; i < class_no; i++){
				if (abandonment_list[i].size() != 1) {
					min_temp[i] = *min_element(abandonment_list[i].begin()+1, abandonment_list[i].end());
				} 
			}

			t_abandon = *min_element(min_temp.begin(), min_temp.end());
			
			//find the class of the customer who will abandon next 
			for (int i = 0; i < class_no; i++){
				auto itr = find(abandonment_list[i].begin(), abandonment_list[i].end(), t_abandon);
				if (itr != abandonment_list[i].end()){
					class_abandon = i;
					break;
				}
			}
			
			auto cust_itr = find(abandonment_list[class_abandon].begin() + 1, abandonment_list[class_abandon].end(), t_abandon); 
			cust_abandon = distance(abandonment_list[class_abandon].begin() + 1, cust_itr); 
		}
		else { // No one in the queue
			t_abandon = std::numeric_limits<double>::infinity(); // No more abandonment events
		}

		// Handle preemptive scheduling if the system is over capacity or intervals differ
		if ((num_in_system[0] > no_server[interval] or pre_interval != post_interval)) {

			std::vector<double> priority_order;
			int opt_policy;
			
			priority_order = queueing_discipline(num_in_system, interval);
			
			// Adjust service and queue states based on priority
			int avail_server_num; 	// number of available agents
			int num_served = 0; 	// number of customers being served
			std::vector<int> optimal_policy(class_no, 0);

			for (int i = 0; i < priority_order.size(); i++){
				int ind = priority_order[i]; 													// Class based on priority order
				avail_server_num = std::max(no_server[interval] - num_served, 0);				// Number of agents available 
				optimal_policy[ind] = std::max(num_in_system[ind + 1] - avail_server_num, 0);	// The optimal number of people from this class in the queue (Number of people in the system from that class - number of available)
				num_served += std::min(num_in_system[ind + 1], avail_server_num);				// The number of customers being served is equal to minimum of number of people in the system and number of available servers
			}

			int* diff = new int[class_no];

			// Calculate the difference between current queue length and the optimal queue length
			for (int i = 0; i < class_no; i++){
				diff[i] = num_in_queue[i+1] - optimal_policy[i]; // difference -- if positive, people should not be placed in the queue and should be send to service; if negative, more people should be placed in the queue.
				num_in_service[i + 1] += diff[i]; // 
				num_in_queue[0] -= diff[i];
				num_in_queue[i + 1] -= diff[i];

				if (diff[i] >= 0){
					// Admit customers to service from class i 
					for (int j = 0; j < diff[i]; j++){
						remove_queue(i); // remove them from their corresponding queue
					}
				} else if (diff[i] < 0){
					// Add customers back to the queue
					for (int j = 0; j < abs(diff[i]); j++){
						add_queue(t_event,i);
					}
				}
			}

			// Update total number of people in service
     	 	num_in_service[0] = 0;
     	 	for (int i = 0; i < class_no; i++){
     	 		num_in_service[0] += num_in_service[i + 1];
     	 	}
			
			// Schedule the next departure
			if (num_in_service[0] == 0) {
    			t_depart = std::numeric_limits<double>::infinity();
			} else {
    			t_depart = sim_clock + generate_service();
			}
			delete[] diff;
		}
	}

	// Auxiliary argsort function 
	std::vector<double> Execute::argsort(const std::vector<double> &array) {
		// Create a vector of indices from 0 to array.size() - 1
		std::vector<double> indices(array.size());
	    std::iota(indices.begin(), indices.end(), 0);

		// Sort indices based on the values in the array
	    std::sort(indices.begin(), indices.end(),
	              [&array](int left, int right) -> bool {
	                  return array[left] < array[right];
	              });
	    return indices;
	}

	void Execute::add_queue(double& arr_time, int& cls){

		std::vector<double> min_temp(class_no, std::numeric_limits<double>::max());
		queue_list[cls].push_back(arr_time);
		abandonment_list[cls].push_back((arr_time + generate_abandon(cls))); 
		
		
		// Find the minimum abandonment time from all the queues 
		for (int i = 0; i < class_no; i++){
			if (abandonment_list[i].size() != 1) { 
				min_temp[i] = *min_element(abandonment_list[i].begin(), abandonment_list[i].end());	
			} 	
		}

		t_abandon = *min_element(min_temp.begin(), min_temp.end());

		// Find the class of the customer who will abandon next 
		for (int i = 0; i < class_no; i++){
				auto itr = find(abandonment_list[i].begin(), abandonment_list[i].end(), t_abandon);
				if (itr != abandonment_list[i].end()){
					class_abandon = i;
					break;
			}
		}

		auto cust_itr = find(abandonment_list[class_abandon].begin() + 1, abandonment_list[class_abandon].end(), t_abandon); 
		cust_abandon = distance(abandonment_list[class_abandon].begin() + 1, cust_itr);
	}


	void Execute::remove_queue(int& cls){
		
		queue_list[cls].pop_front();
		abandonment_list[cls].erase(abandonment_list[cls].begin() + 1);
		std::vector<double> min_temp(class_no, std::numeric_limits<double>::max());
		
		if (num_in_queue[0] > 0){
			for (int i = 0; i < class_no; i++){
				if (abandonment_list[i].size() != 1) { 
					min_temp[i] = *min_element(abandonment_list[i].begin(), abandonment_list[i].end());
				}
			}

			t_abandon = *min_element(min_temp.begin(), min_temp.begin() + min_temp.size());

			// Find the class of the customer who will abandon next 
			for (int i = 0; i < class_no; i++){
				auto itr = find(abandonment_list[i].begin(), abandonment_list[i].end(), t_abandon);
				if (itr != abandonment_list[i].end()){
					class_abandon = i;
					break;
				}
			}
			auto cust_itr = find(abandonment_list[class_abandon].begin() + 1, abandonment_list[class_abandon].end(), t_abandon); 
			cust_abandon = distance(abandonment_list[class_abandon].begin() + 1, cust_itr);
		} else {
			t_abandon = std::numeric_limits<double>::infinity();
		}
	}

	std::vector<double> Execute::run()
	{	
    	
		// Initialize the state variables
		num_in_system.assign(class_no + 1, 0); //0th index is used for the sum
		num_in_service.assign(class_no + 1, 0); //0th index is used for the sum
		num_arrivals.assign(class_no + 1, 0); //0th index is used for the sum
		num_in_queue.assign(class_no + 1, 0); //0th index is used for the sum
		num_abandons.assign(class_no + 1, 0); //0th index is used for the sum
		queue_integral.assign(class_no + 1, 0); //0th index is used for the sum
		service_integral.assign(class_no + 1, 0); //0th index is used for the sum
		system_integral.assign(class_no + 1, 0); //0th index is used for the sum
		holding_cost.assign(class_no, 0); 
		waiting_cost.assign(class_no, 0); 
		num_departs.assign(class_no, 0);
		total_cost = 0;
		
		// Initial state: All servers are busy
		num_in_system[1] = 12;
		num_in_system[2] = 12;
		num_in_system[3] = 12;
        num_in_system[0] = 36;
        
		num_in_service[1] = 12;
		num_in_service[2] = 12;
		num_in_service[3] = 12;
        num_in_service[0] = 36;
		
		interval = 0;
		sim_clock = 0;

		// Schedule the first arrival
		t_arrival = generate_interarrival(interval); //first event should be an arrival
		t_depart = std::numeric_limits<double>::infinity();
		t_abandon = std::numeric_limits<double>::infinity();
		
		// Simulation loop
		while (sim_clock < T){
			// Determine the next event time
			t_event = std::min({t_arrival, t_depart, t_abandon}); 

			// Update integrals based on elapsed time
			double elapsed_time = t_event - sim_clock;
			for (int i = 0; i < class_no + 1; i++) {
				queue_integral[i] += num_in_queue[i] * elapsed_time;
				service_integral[i] += num_in_service[i] * elapsed_time;
				system_integral[i] += num_in_system[i] * elapsed_time;
			}
			
			// Update costs
			for (int i = 0; i < class_no; i++){
				holding_cost[i] += num_in_queue[i + 1] * holding_cost_rate[i] * elapsed_time; 
				waiting_cost[i] += num_in_queue[i + 1] * cost_rate[i] * elapsed_time;
				total_cost += num_in_queue[i + 1] * cost_rate[i] * elapsed_time;
			}  

			// Advance simulation clock
			sim_clock = t_event; 
			pre_interval = interval;
			interval = std::min(int(sim_clock*12),203);
			post_interval = interval;

			// Handle the event
			if (t_event == t_arrival) { // Arrival event 
				//std::cout << " arrival " << std::endl;
				std::uniform_real_distribution<double> uniform(0.0, 1.0); 										// Look up seed
            	double arrival_seed = uniform(generator);						
            	auto low = std::lower_bound(arr_cdf[interval].begin(), arr_cdf[interval].end(), arrival_seed);  // Determine the class that has arrived based on the seed
				int arrival_ind = low - arr_cdf[interval].begin(); 
            	handle_arrival_event(interval, arrival_ind, pre_interval, post_interval);
			
			}  else if (t_event == t_depart) { // Departure event 
				//std::cout << " depart " << std::endl;
				std::uniform_real_distribution<double> uniform(0.0, 1.0); 										// Look up seed 
            	double departure_seed = uniform(generator);

				std::vector<double> numerator(class_no, 0);
				
				for (int i = 0; i < class_no; i++){
					numerator[i] = num_in_service[i + 1] * mu_hourly[i]; 										// Total service rate for each class based on the number of people in service from that class
				}
				double initial_sum = 0;
				double total_service_rate = std::accumulate(numerator.begin(), numerator.end(), initial_sum);
				std::vector<double> service_cdf;

				double cumulative = 0.0;
				for (const auto& rate: numerator){
					cumulative += rate / total_service_rate;
					service_cdf.push_back(cumulative);
				}
		
				auto low = std::lower_bound(service_cdf.begin(), service_cdf.end(), departure_seed); 			// Determine the class that has departed based on the seed
				int service_ind = low - service_cdf.begin();
				handle_depart_event(interval, service_ind, pre_interval, post_interval);				
			} else if (t_event == t_abandon) { // Abandonment event 
				//std::cout << " abandon " << std::endl;
				handle_abandon_event(interval, pre_interval, post_interval);
			} else {std::cout << "Something is Wrong" << std::endl;}
		   	
		    //std::cout << "num_in_system: " <<  num_in_system[0] << " | num_in_queue: " <<  num_in_queue[0] << " | num_in_service: " <<  num_in_service[0] <<std::endl;
 		}

		double overtime_cost = 2.12;
		std::vector<double> res(class_no + 1, 0);

		// Cost for each class at the end of the simulation
		for (int i = 0; i < class_no; i ++){
			res[i] = waiting_cost[i] + overtime_cost * num_in_queue[i + 1]; 	// Adding the overtime cost to the waiting cost for each class
		}
		// Total cost at the end of the simulation
		res[class_no] = total_cost + overtime_cost * num_in_queue[0]; 			// Adding the overtime cost to the total cost
       	return res;
	}
}

int main(int argc, char** argv){ 

	simulation::Simulation simObj;// = new simulation::Simulation(); 	
	
	simObj.save();

	return 0;
}
