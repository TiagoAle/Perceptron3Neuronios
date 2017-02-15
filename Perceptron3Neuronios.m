clc;
clear all;
close all;

B = load('iris.data');
B = B(randperm(size(B,1)),:);

treinamento = B(1:120,:);
teste = B(121:150,:);

nPadroes = 120;
nEntradas = 5;

txAprendizado = 0.1;
w = zeros(5,3);
bias = repmat(-1,nPadroes,1);

%X = randperm(X,1);

treinamento = [bias treinamento];
teste = [bias(1:30) teste];

erroEpoca = 0;
epoca = 0;
Y = zeros(nPadroes,3);

while(epoca < 100)
  %Y = zeros(nPadroes,3);
  D = treinamento(:,6:8);
  X = treinamento(:,1:5);
  erroEpoca = 0;
  
  for j = 1:nPadroes
    u = X(j,:)*w;
    
    for i = 1:3
        if(u(:,i)> 0)
           Y(j,i) = 1;
        else
          Y(j,i) = 0;
        end
    end
    
    erro = D-Y;
    erro(j,:)
    w = w +txAprendizado*X(j,:)'*erro(j,:);
    
    erroEpoca = sum(sum(abs(erro)));
  end
  epoca = epoca + 1;
  if (erroEpoca < 5)
      break;
  end
  %treinamento = treinamento(randperm(size(treinamento,1)),:);
end

disp('Treinamento do Perceptron (AND)');
disp('Resultados do Treinamento');
disp(' ');
disp(['Épocas: ',num2str(epoca)]);
disp(['w0:   ',num2str(w(1,:))]);
disp(['w1:     ',num2str(w(2,:))]);
disp(['w2:     ',num2str(w(3,:))]);
disp(['w3:     ',num2str(w(4,:))]);
disp(['w4:     ',num2str(w(5,:))]);
disp(' ');

    