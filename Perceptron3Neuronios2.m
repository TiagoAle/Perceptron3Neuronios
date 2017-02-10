clc;
clear all;
close all;

B = load('iris.data');

D = B(:,5:7);
X = B(:,1:4);

[nPadroes, nEntradas] = size(X);

txAprendizado = 0.1;
w = rand(nEntradas+1,3);
bias = repmat(-1,nPadroes,1);
minX = repmat(min(X),nPadroes,1);
maxX = repmat(max(X),nPadroes,1);
X = (X - minX)./(maxX - minX);
X = [bias X];

erroEpoca = 31;
epoca = 0;
Y = zeros(nPadroes,3);

while(epoca < 100)
  
  erroEpoca = 0;
  
  for j = 1:nPadroes
    u = w'*X(j,:)';
    u = u';
    
    if(u(:,1)> u(:,2) && u(:,1) > u(:,3))
        Y(j,1) = 1;
        Y(j,2) = 0;
        Y(j,3) = 0;
    elseif (u(:,2)> u(:,3))
        Y(j,1) = 0;
        Y(j,2) = 1;
        Y(j,3) = 0;
    else
        Y(j,1) = 0;
        Y(j,2) = 0;
        Y(j,3) = 1;
    end
    
    erro = D-Y;
    w = w +txAprendizado*X'*erro;
    erroEpoca = sum(sum(abs(erro)));
  end
  if(erroEpoca < 5)
      break;
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

    