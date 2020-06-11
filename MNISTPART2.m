clear all;
close all;
clf;
clc;

number_of_hidden_layer = 1;                         
hidden_neurons = 150;                              
output_neurons = 784;                             

train_data = load('MNISTtraindata.txt');   
train_label = load('MNISTtrainlabel.txt');
test_data = load('MNISTtestdata.txt');
test_label = load('MNISTtestlabel.txt');
                                
epochs = 100;                              
eta_1 = 0.01;                           
eta_2 = 0.01;                           
alpha = 0.5;                           
J2_loss_train = [];                         
J2_loss_test = [];                           
J2_epoch_train(1:epochs+1) = (0);  

weight_input = normrnd(0, sqrt(6/(hidden_neurons+length(train_data(1,:)))), [hidden_neurons, length(train_data(1,:))+1]); 
weight_output = normrnd(0, sqrt(6/(hidden_neurons+output_neurons)), [output_neurons, hidden_neurons+1]); 

delta_layer2(1,1:output_neurons) = (0);                  
delta_weight_output(1:output_neurons, 1:hidden_neurons+1) = (0);           
d_weight_input(1:hidden_neurons, 1:length(train_data(1,:))+1) = (0); 

% Training 
for k = 1:epochs+1
    for i=1:length(train_data)              
        t_label = train_label(i);             
        actual_output = train_data(i,:);        
        layer1_neu_sum = (weight_input*([1 train_data(i,:)]'))';     
        layer1_op = 1./(1+exp(-layer1_neu_sum));            
        layer2 = (weight_output*([1 layer1_op]'))';          
        layer2_output = 1./(1+exp(-layer2));            
        
        delta_layer2 = (actual_output - layer2_output).*layer2_output.*(1-layer2_output);   
        delta_weight_output = eta_2*delta_layer2'*([1 layer1_op]) + alpha*delta_weight_output;    
        weight_output = weight_output + delta_weight_output;                                   
        
        delta_layer1 = weight_output(1:output_neurons,2:hidden_neurons+1)'*delta_layer2';              
        delta_lay1 = layer1_op.*(1-layer1_op).*delta_layer1';             
        d_weight_input =  eta_1*delta_lay1'*([1 train_data(i,:)]) + alpha*d_weight_input;  
        weight_input = weight_input + d_weight_input ;                                  
        
        error_matrix = (actual_output-layer2_output).^2;                   
        error_e = 0.5*sum(error_matrix);
        J2_epoch_train(k) = J2_epoch_train(k) + error_e;
        
        if k == epochs
            error_matrix = (actual_output-layer2_output).^2;
            error = 0.5*sum(error_matrix);
            J2_loss_train(t_label+1,end+1) = error;
        end
    end
end

% Testing the Neural Network
for test = 1:length(test_data)                                     
    tst_label = test_label(test);                                   
    actual_output = test_data(test,:);                              
    
    layer1_neu_sum = (weight_input*([1 test_data(test,:)]'))';              
    layer1_op = 1./(1+exp(-layer1_neu_sum));                        

    layer2 = (weight_output*([1 layer1_op]'))';                      
    layer2_output = 1./(1+exp(-layer2));                        
    
    error_matrix = (actual_output-layer2_output).^2;                         
    error = 0.5*sum(error_matrix);
    J2_loss_test(tst_label+1,end+1) = error;
end


train_number = sum(J2_loss_train~=0,2);
train_sum = sum(J2_loss_train,2);
mean_J2train_num = train_sum./train_number;
mean_J2train_all = sum(train_sum)/sum(train_number);

test_num = sum(J2_loss_test~=0,2);
test_sum = sum(J2_loss_test,2);
mean_J2test_num = test_sum./test_num;
mean_J2test_all = sum(test_sum)/sum(test_num);

% Plot the J2 loss function values of the neural network every 10th epoch
figure(1)
Graph1 = categorical({'Mean Loss on Training set','Mean Loss on Test set'});
mean_loss = [mean_J2train_all; mean_J2test_all];
bar(Graph1, mean_loss);
title('Mean Loss on Training and Test Set');
ylabel('J2 Mean Loss Values');

% Plot the mean J2 loss function values of the neural network in training and testing data set
figure(3)
J2_train_plot_x = [0 10:10:epochs];
J2_train_plot = [J2_epoch_train(1)/4000 (J2_epoch_train(11:10:epochs+1)/4000)];
plot(J2_train_plot_x, J2_train_plot, 'bo-');
title('Mean Loss on Training Set vs. Epochs');
xlabel('Epochs');
ylabel('Mean J2 values');
xticks([0 10:10:epochs]);

% numbers in training and testing data set
figure(2) 
Graph2_labels = categorical({'Mean Loss 0','Mean Loss 1','Mean Loss 2','Mean Loss 3','Mean Loss 4','Mean Loss 5','Mean Loss 6','Mean Loss 7','Mean Loss 8','Mean Loss 9'});
Loss_num_mean = [mean_J2train_num(1) mean_J2test_num(1); mean_J2train_num(2) mean_J2test_num(2); mean_J2train_num(3) mean_J2test_num(3); mean_J2train_num(4) mean_J2test_num(4); mean_J2train_num(5) mean_J2test_num(5); mean_J2train_num(6) mean_J2test_num(6); mean_J2train_num(7) mean_J2test_num(7); mean_J2train_num(8) mean_J2test_num(8); mean_J2train_num(9) mean_J2test_num(9); mean_J2train_num(10) mean_J2test_num(10)];
bar(Graph2_labels, Loss_num_mean);
title('Mean Loss on Training and Test Set for each number');
ylabel('J2 Mean Loss Values');
legend ('Training set','Test set');

% Code reference Dr. Ali Minai's code 
figure(4)
random_hidden_neuron = randperm(150,20);
dlmwrite('20RandomHiddenNeuronNum.txt', random_hidden_neuron);
A = weight_input(random_hidden_neuron, 2:785);
for i=1:4
    for j = 1:5
        v = reshape(A((i-1)*5+j,:),28,28);
        subplot(4,5,(i-1)*5+j)
        image(64*v) 
        colormap(gray(64));
        set(gca,'xtick',[])
        set(gca,'xticklabel',[])
        set(gca,'ytick',[])
        set(gca,'yticklabel',[])
        set(gca,'dataaspectratio',[1 1 1]);
    end
end
a1 = axes; 
t1 = title('Images of 20 Random Hidden Neuron Weights');
a1.Visible = 'off';
t1.Visible = 'on';

% Code reference: Dr. Ali Minai's code 
figure(5)
P1 = load('IPweights.txt');
A = P1(random_hidden_neuron, 2:785);
for i=1:4
    for j = 1:5
        v = reshape(A((i-1)*5+j,:),28,28);
        subplot(4,5,(i-1)*5+j)
        image(64*v) 
        colormap(gray(64));
        set(gca,'xtick',[])
        set(gca,'xticklabel',[])
        set(gca,'ytick',[])
        set(gca,'yticklabel',[])
        set(gca,'dataaspectratio',[1 1 1]);
    end
end
a2 = axes; 
t2 = title('Images of Problem 1 Corresponding 20 Random Hidden Neuron Weights from P2');
a2.Visible = 'off';
t2.Visible = 'on';

% Code reference: Dr. Ali Minai's code 
figure(6)
image_1 = [];
r_test_data = randperm(1000,8);
orginal_image = test_data(r_test_data, :);
for t = 1:length(r_test_data)   
    layer1_neu_sum = (weight_input*([1 orginal_image(t,:)]'))';
    layer1_op = 1./(1+exp(-layer1_neu_sum));

    layer2 = (weight_output*([1 layer1_op]'))';
    layer2_output = 1./(1+exp(-layer2));
    image_1(t, :) = layer2_output;
end
for i=1:2
    if i==1
        A = orginal_image;
    else
        A = image_1;
    end
    for j = 1:8
        v = reshape(A(j,:),28,28);
        subplot(2,8,(i-1)*8+j)
        image(64*v) 
        colormap(gray(64));
        set(gca,'xtick',[])
        set(gca,'xticklabel',[])
        set(gca,'ytick',[])
        set(gca,'yticklabel',[])
        set(gca,'dataaspectratio',[1 1 1]);
    end
end
a3 = axes; 
t3 = title('Original Image vs. Reconstructed Image');
a3.Visible = 'off';
t3.Visible = 'on';


dlmwrite('Trained_IPweights.txt', weight_input);
dlmwrite('Trained_OPweights.txt', weight_output);
