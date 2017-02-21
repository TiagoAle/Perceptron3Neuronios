clc;
clear all;
close all;
%Carregando dados da iris embaralhados
B = load('iris.data');
B = B(randperm(size(B,1)),:);
%Separando amostras para treinamento e testes
treinamento = B(1:120,:);
teste = B(121:150,:);

nPadroes = 120;
nEntradas = 5;
%Definindo valores iniciais de pesos, taxa de apredizado e bias
txAprendizado = 0.1;
w = zeros(5,3);
bias = repmat(-1,nPadroes,1);

%X = randperm(X,1);
%Adicionando o bias a matriz de treinamento e testes
treinamento = [bias treinamento];
teste = [bias(1:30) teste];

erroEpoca = 0;
epoca = 0;
Y = zeros(nPadroes,3);
%Separando os valores de entradas X dos valores desejados D
D = treinamento(:,6:8);
X = treinamento(:,1:5);
%Incianando o treinamento
while(epoca < 100)
  %Y = zeros(nPadroes,3);
  erroEpoca = 0;
  
  for j = 1:nPadroes
    u = X(j,:)*w;
    %Função sinal para cada neuronio
    for i = 1:3
        if(u(:,i)> 0)
           Y(j,i) = 1;
        else
          Y(j,i) = 0;
        end
    end
    %Detecção de erro e atualização dos pesos
    erro = D(j,:)-Y(j,:);
    w = w +txAprendizado*X(j,:)'*erro;
    %Contagem do erro da época
    if (sum(abs(erro)))
        erroEpoca = erroEpoca + 1;
    end
  end
  epoca = epoca + 1;
  %Outra condição de parada caso o erroEpoca esteja tolerável
  if (erroEpoca < 5)
      break;
  end
  %treinamento = treinamento(randperm(size(treinamento,1)),:);
end

%Iniciando os testes
Dteste = teste(:,6:8);
Xteste = teste(:,1:5);
erroTeste = 0;
YTeste = zeros(30,3);
for j = 1:30
    u = Xteste(j,:)*w;
    for i = 1:3
        if(u(:,i)> 0)
           YTeste(j,i) = 1;
        else
          YTeste(j,i) = 0;
        end
    end
    if(sum(u)> 1)
       y = sigmf(u, [1 0]);
       YTeste(j,:)
       if y(1) > y(2) && y(1) > y(3)
           YTeste(j,:) = [1 0 0];
       elseif y(2) > y(3)
           YTeste(j,:) = [0 1 0];
       else
           YTeste(j,:) = [0 0 1];
       end
       YTeste(j,:)
    end
    error(j,:) = Dteste(j,:)-YTeste(j,:);
    e = sum(abs(error(j,:)));
    if(e ~= 0)
        erroTeste = erroTeste + 1;
    end 
end
%Print dos valores de pesos e épocas durante o algoritmo 
disp(erroTeste);
disp('Treinamento do Perceptron');
disp('Resultados do Treinamento');
disp(' ');
disp(['Épocas: ',num2str(epoca)]);
disp(['w0:   ',num2str(w(1,:))]);
disp(['w1:     ',num2str(w(2,:))]);
disp(['w2:     ',num2str(w(3,:))]);
disp(['w3:     ',num2str(w(4,:))]);
disp(['w4:     ',num2str(w(5,:))]);
disp(' ');

    