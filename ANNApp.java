public class ANNApp {

	public static void main(String[] args) {

		int iteration = 1; //modify the number of iteration
		
		int input_node = 2;
		int hidden_node = 2;
		int output_node = 2;
		double learning_rate = 0.5;
		double[] bias = { 0.35, 0.6 };

		double[] input = { 0.05, 0.1 };
		double[] target = { 0.01, 0.99 };
		double[] weights_hidden_layer = { 0.15, 0.2, 0.25, 0.3 };
		double[] weights_output_layer = { 0.4, 0.45, 0.5, 0.55 };

		Ann_Engine ann = new Ann_Engine(input_node, hidden_node, output_node, 
				learning_rate, bias, input, target, weights_hidden_layer, 
				weights_output_layer);
		
		for (int i = 0; i < iteration; i++) {
			ann.updateWeight(ann.weights_hidden_layer_plus, ann.weights_output_layer_plus);
			ann.train();
		}
		//keep latest weight
		ann.updateWeight(ann.weights_hidden_layer_plus, ann.weights_output_layer_plus);
		
		//test engine
		double[] test_input = { 0.05, 0.1 };
		ann.test(test_input);
	}
}
