function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));



%feed forward propagation
X=[ones(m,1) X];
a1=X*Theta1'; e=a1;
a1=sigmoid(a1);
a1=[ones(m,1),a1];
a2=a1*Theta2'; 
a2=sigmoid(a2);

for i=1:m
   r=zeros(num_labels,1);
  h=a2(i,:); j=y(i); r(j)=1;
  h1=log(h); h2=log(1-h);
  c=h1*r +h2*(1-r);
  J=J+c;
endfor
J=(-1/m)*J;
t1=Theta1; t2=Theta2;
t1(:,1)=0; t2(:,1)=0;
t1=t1.^2; t2=t2.^2;
t1=sum(sum(t1)); t2=sum(sum(t2));
t=(lambda/(2*m))*(t1+t2);
%cost with reqularization
J=J+t;
%backpropagation part
A1=X;
z2=A1*Theta1';
A2=sigmoid(z2);
A2=[ones(size(A2,1),1),A2];
z3=A2*Theta2';
A3=sigmoid(z3); %5000*10
yvec=(1:num_labels)==y;  %5000*10
delta3=A3-yvec;
delta2=(delta3 *Theta2).*[ones(size(z2,1),1) sigmoidGradient(z2)];  %sigmoidGradient(z2) -->5000*25
delta2=delta2(:,2:end);
Theta1_grad=(1/m)*(delta2'*A1);
Theta2_grad=(1/m)*(delta3'*A2);
Theta1_grad_reg=(lambda/m)*[zeros(size(Theta1,1),1)  Theta1(:,2:end)];
Theta2_grad_reg=(lambda/m)*[zeros(size(Theta2,1),1)  Theta2(:,2:end)];
Theta1_grad=Theta1_grad +Theta1_grad_reg;
Theta2_grad=Theta2_grad +Theta2_grad_reg;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
