clc;
clear all;
close all;

B = load('iris.data');

D = B(:,5:7);
X = B(:,1:4);

[nPadroes, nEntradas] = size(X);

txAprendizado = 0.5;
w = rand(nEntradas+1,3);
bias = repmat(-1,nPadroes,1);
minX = repmat(min(X),nPadroes,1);
maxX = repmat(max(X),nPadroes,1);
X = (X - minX)./(maxX - minX);
X = [bias X];

erroEpoca = 31;
epoca = 0;
Y = zeros(nPadroes,3);

while(epoca < 23)
  
  erroEpoca = 0;
  
  for j = 1:nPadroes
    u = w'*X(j,:)';
    u = u';
    
    for i = 1:3
        tangenteH = tanh(u(:,i));
        if(tangenteH > 0)
           Y(j,i) = 1;
        else
          Y(j,i) = 0;
        end
    end
    
    erro = D-Y;
    w = w +txAprendizado*X'*erro;
    erroEpoca = sum(sum(abs(erro)));
  end
  epoca = epoca + 1;
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
disp('Teste para os pesos encontrados: ')
for j=1:4
    disp(['X1 = ',num2str(X(j,1)),' X2 = ',num2str(X(j,2)),...
        ' Yobtido = ',num2str(Y(j)),' Ydesejado = ',num2str(D(j))]);
end

    