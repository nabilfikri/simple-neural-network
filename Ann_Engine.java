public class Ann_Engine {

	int input_node;
	int hidden_node;
	int output_node;
	double learning_rate;
	double bias[];
	double input[];
	double target[];
	double weights_hidden_layer[];
	double weights_output_layer[];

	double weights_hidden_layer_plus[];
	double weights_output_layer_plus[];
	double hidden_net[];
	double hidden_out[];
	double output_net[];
	double output_out[];

	public Ann_Engine() {

	}

	public Ann_Engine(int input_node, int hidden_node, int output_node, double learning_rate, double[] bias,
			double[] input, double[] target, double[] weights_hidden_layer, double[] weights_output_layer) {
		super();
		this.input_node = input_node;
		this.hidden_node = hidden_node;
		this.output_node = output_node;
		this.learning_rate = learning_rate;
		this.bias = bias;
		this.input = input;
		this.target = target;
		this.weights_hidden_layer = weights_hidden_layer;
		this.weights_output_layer = weights_output_layer;

		weights_hidden_layer_plus = weights_hidden_layer.clone();
		weights_output_layer_plus = weights_output_layer.clone();

		hidden_net = new double[hidden_node];
		hidden_out = new double[hidden_node];
		output_net = new double[output_node];
		output_out = new double[output_node];
	}

	public void updateWeight(double weight_hidden_plus[], double[] weight_output_plus) {
		weights_hidden_layer = weight_hidden_plus.clone();
		weights_output_layer = weight_output_plus.clone();
	}

	public void train() {
		//calculate hidden net and hidden out
		int weight_count = 0;
		for (int i = 0; i < hidden_node; i++) {
			double sum = 0.0;
			for (int j = 0; j < input_node; j++) {
				sum += input[j] * weights_hidden_layer[weight_count++];
			}
			hidden_net[i] = sum + bias[0] * 1;
			hidden_out[i] = fn_logistic(hidden_net[i]);
		}

		//calculate output net and output out
		weight_count = 0;
		for (int i = 0; i < output_node; i++) {
			double sum = 0.0;
			for (int j = 0; j < hidden_node; j++) {
				sum += hidden_out[j] * weights_output_layer[weight_count++];
			}
			output_net[i] = sum + bias[1] * 1;
			output_out[i] = fn_logistic(output_net[i]);
		}

		//calculate total error
		double error_total = 0.0;
		for (int i = 0; i < output_node; i++) {
			error_total += calc_error(target[i], output_out[i]);
		}
		System.out.println("Total Error: " + error_total);

		//backward pass (from output to hidden)
		//apply chain rule

		double[] d_total_error_wrt_weight = new double[weights_output_layer.length]; 
		double[] d_total_error_wrt_output_out = new double[weights_output_layer.length]; 
		double[] d_output_out_wrt_output_net = new double[weights_output_layer.length]; 
		double[] d_output_net_wrt_weight = new double[weights_output_layer.length]; 

		weight_count = 0;
		for (int i = 0; i < output_node; i++) {
			for (int j = 0; j < hidden_node; j++) {
				d_total_error_wrt_output_out[weight_count] = -(target[i] - output_out[i]);
				d_output_out_wrt_output_net[weight_count] = output_out[i] * (1 - output_out[i]);
				d_output_net_wrt_weight[weight_count] = hidden_out[j];
				d_total_error_wrt_weight[weight_count] = d_total_error_wrt_output_out[weight_count] * d_output_out_wrt_output_net[weight_count] * d_output_net_wrt_weight[weight_count];

				weights_output_layer_plus[weight_count] = weights_output_layer[weight_count] - learning_rate * d_total_error_wrt_weight[weight_count];
				weight_count++;
			}
		}
//		int a = 0;
//		for (double d : weights_output_layer_plus) {
//			System.out.println("w" + a++ + "+ : " + d);
//		}

		//preparation: generate indices for backward pass (connection (weight) to backward - from output to hidden)
		int[] backward_weight_index = new int[weights_output_layer.length];
		int index = 0;
		weight_count = 0;
		for (int i = 0; i < hidden_node; i++) {
			index = 0;
			for (int j = 0; j < output_node; j++) {
				backward_weight_index[weight_count++] = index + i;
				index += hidden_node;
			}
		}

//		for (int d : backward_weight_index) {
//			System.out.println(d);
//		}
//		System.exit(0);

		//backward pass (from hidden to input)
		//apply chain rule with bit of adjustment		
		double[] d_total_err_wrt_weight = new double[weights_hidden_layer.length];		//0
		double[] d_total_err_wrt_hidden_out = new double[hidden_node];					//1
		double[] d_hidden_out_wrt_hidden_net = new double[hidden_node];					//2
		//double[] d_hidden_net_wrt_weight = new double[weights_hidden_layer.length];	//3, skipped

		//decompose 1, calc 1
		weight_count = 0;
		double sum;
		for (int i = 0; i < hidden_node; i++) {
			sum = 0;
			for (int j = 0; j < output_node; j++) {
				double output_err_wrt_output_out = (-(target[j] - output_out[j]));
				double output_out_wrt_output_net = (output_out[j] * (1 - output_out[j]));
				double d_output_err_wrt_output_net = output_err_wrt_output_out * output_out_wrt_output_net;		//a1
				double d_output_net_wrt_hidden_out = weights_output_layer[backward_weight_index[weight_count]];	//a2
				double d_output_err_wrt_hidden_out = d_output_err_wrt_output_net * d_output_net_wrt_hidden_out;
				sum += d_output_err_wrt_hidden_out;
				weight_count++;
			}
			d_total_err_wrt_hidden_out[i] = sum;
		}

		//calc 2
		for (int i = 0; i < hidden_node; i++) {
			d_hidden_out_wrt_hidden_net[i] = hidden_out[i] * (1 - hidden_out[i]);
		}

		//calc 3
		//skip

		//calculate 
		weight_count = 0;
		for (int i = 0; i < hidden_node; i++) {
			for (int j = 0; j < input_node; j++) {
				d_total_err_wrt_weight[weight_count] = d_total_err_wrt_hidden_out[i] * d_hidden_out_wrt_hidden_net[i] * input[j];
				weight_count++;
			}
		}

		//update new weight
		for (int i = 0; i < weights_hidden_layer.length; i++) {
			weights_hidden_layer_plus[i] = weights_hidden_layer[i] - learning_rate * d_total_err_wrt_weight[i];
		}

//		for (double d : weights_hidden_layer_plus) {
//			System.out.println(d);
//		}
	}

	public void test(double[] test_input) {

		double test_hidden_net[] = new double[hidden_node];
		double test_hidden_out[] = new double[hidden_node];
		double test_output_net[] = new double[output_node];
		double test_output_out[] = new double[output_node];

		//calculate hidden net and hidden out
		int weight_count = 0;
		for (int i = 0; i < hidden_node; i++) {
			double sum = 0.0;
			for (int j = 0; j < input_node; j++) {
				sum += test_input[j] * weights_hidden_layer[weight_count++];
			}
			test_hidden_net[i] = sum + bias[0] * 1;
			test_hidden_out[i] = fn_logistic(test_hidden_net[i]);
		}

		//calculate output net and output out
		weight_count = 0;
		for (int i = 0; i < output_node; i++) {
			double sum = 0.0;
			for (int j = 0; j < hidden_node; j++) {
				sum += test_hidden_out[j] * weights_output_layer[weight_count++];
			}
			test_output_net[i] = sum + bias[1] * 1;
			test_output_out[i] = fn_logistic(test_output_net[i]);
		}

		System.out.println("Result from test");
		for (double d : test_output_out) {
			System.out.println(d);
		}
	}

	private static double calc_error(double target, double out) {
		return 0.5 * Math.pow((target - out), 2);
	}

	private static double fn_logistic(double hidden_out) {
		return 1 / (1 + Math.exp(-hidden_out));
	}
}
