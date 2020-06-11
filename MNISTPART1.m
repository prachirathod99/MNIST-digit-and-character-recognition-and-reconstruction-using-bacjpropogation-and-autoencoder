close all 
clear all
clc

%MNIST dataset Part 1

input_layer = 'Input the number of required hidden layers: ';               
hidden_layer = input(input_layer);                 
input_neuron = 'Input the number of hidden neurons to be implemented: ';    
hidden_neurons = input(input_neuron);                                      
output_neurons = 10;                                                   

imgage_data = load('MNISTnumImages5000.txt'); %loads the image data
image_label = load('MNISTnumLabels5000.txt'); %loads the labels

random_number = randperm(size(imgage_data,1)); %randomizing the data
random_order_data = imgage_data(random_number,:);
random_order_lab = image_label(random_number,:);

epochs = 100;                                  
eta_1 = 0.1;                                
eta_2 = 0.1;                                
alpha = 0.5;       

confusion_train_matrix(1:10,1:10) = (0);                
confusion_test_matrix(1:10,1:10) = (0);  

%spliting the datat into test and train
train_data = random_order_data(1:4000,:);
train_lab = random_order_lab(1:4000,:);
test_data = random_order_data(4001:5000,:);
test_lab = random_order_lab(4001:5000,:);

%saving the data

dlmwrite('MNISTtraindata.txt', train_data);
dlmwrite('MNISTtestdata.txt', test_data);
dlmwrite('MNISTtrainlabel.txt', train_lab);
dlmwrite('MNISTtestlabel.txt', test_lab);

true_train_count(epochs+1,1) = (0);                  
true_test_count = 0;  

%Initializing the weihghts
weight_input = normrnd(0, sqrt(6/(hidden_neurons+length(train_data(1,:)))), [hidden_neurons, length(train_data(1,:))+1]); 
weight_output = normrnd(0, sqrt(6/(hidden_neurons+output_neurons)), [output_neurons, hidden_neurons+1]); 

%initializing the weights to zeors
delta_layer2(1,1:output_neurons) = (0);                      
delta_weight_output(1:output_neurons, 1:hidden_neurons+1) = (0);               
delta_weight_input(1:hidden_neurons, 1:length(train_data(1,:))+1) = (0);  

% Training  
for k = 1:epochs+1
    for i=1:length(train_data)                 
        t_label = train_lab(i);                
        actual_output(1:output_neurons) = (0); 
        actual_output(t_label+1) = 1;          
            
        layer1 = (weight_input*([1 train_data(i,:)]'))';    
        layer1_output = 1./(1+exp(-layer1));            
        
        layer2 = (weight_output*([1 layer1_output]'))';          
        layer2_output = 1./(1+exp(-layer2));            
            
        delta_layer2 = (actual_output - layer2_output).*layer2_output.*(1-layer2_output);   
        delta_weight_output = eta_2*delta_layer2'*([1 layer1_output]) + alpha*delta_weight_output;    
        weight_output = weight_output + delta_weight_output;                                   
        
        d_layer1_sum = weight_output(1:output_neurons,2:hidden_neurons+1)'*delta_layer2';              
        d_layer1 = layer1_output.*(1-layer1_output).*d_layer1_sum';             
        delta_weight_input =  eta_1*d_layer1'*([1 train_data(i,:)]) + alpha*delta_weight_input;  
        weight_input = weight_input + delta_weight_input ;                                  
        
        [x,y] = max(layer2_output);                           
        layer2_output(y) = 1;                                 
        layer2_output(layer2_output<x) = 0;
        if (isequal(layer2_output, actual_output))                
            true_train_count(k) = true_train_count(k) + 1;
        end
        if k==epochs+1                                   
            confusion_train_matrix(find(actual_output == max(actual_output)), find(layer2_output == max(layer2_output))) = confusion_train_matrix(find(actual_output == max(actual_output)), find(layer2_output == max(layer2_output))) + 1;
        end
    end
end

hit_rates_train = true_train_count/length(train_data);
Error_train = 1 - hit_rates_train;
Error_plot_x = [0 10:10:epochs];
Error_train_plot = [Error_train(1) (Error_train(11:10:epochs+1))'];

figure(1);
plot(Error_plot_x, Error_train_plot, 'bo-');
title('Training Error Values of Neural Network vs. Epochs');
xlabel('Epochs');
ylabel('Error values (1-hit-rate)');
xticks([0 10:10:epochs]);

% Testing 
for test = 1:length(test_data)                            
    tst_label = test_lab(test);                           
    actual_output(1:output_neurons) = (0);                             
    actual_output(tst_label+1) = 1;                         
    
    layer1 = (weight_input*([1 test_data(test,:)]'))';      
    layer1_output = 1./(1+exp(-layer1));                
    
    layer2 = (weight_output*([1 layer1_output]'))';      
    layer2_output = 1./(1+exp(-layer2));               
    
    [x,y] = max(layer2_output);                               
    layer2_output(y) = 1;
    layer2_output(layer2_output<x) = 0;
    if (isequal(layer2_output, actual_output))
        true_test_count = true_test_count + 1;
    end                                                 
    confusion_test_matrix(find(actual_output == max(actual_output)), find(layer2_output == max(layer2_output))) = confusion_test_matrix(find(actual_output == max(actual_output)), find(layer2_output == max(layer2_output))) + 1;
end

dlmwrite('IPweights.txt', weight_input);
dlmwrite('OPweights.txt', weight_output);
