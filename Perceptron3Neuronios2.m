clc;
clear all;
close all;

B = load('iris.data');

treinamento = B(1:120,:);
teste = B(121:150,:);

nPadroes = 120;
nEntradas = 5;

txAprendizado = 0.1;
w = rand(nEntradas,3);
bias = repmat(-1,nPadroes,1);
%minX = repmat(min(X),nPadroes,1);
%maxX = repmat(max(X),nPadroes,1);
%X = (X - minX)./(maxX - minX);
treinamento = [bias treinamento];
teste = [bias(1:30) teste];

erroEpoca = 31;
epoca = 0;
Y = zeros(nPadroes,3);

D = treinamento(:,6:8);
X = treinamento(:,1:5);

while(epoca < 100)
  
  erroEpoca = 0;
  
  for j = 1:nPadroes
    u = w'*X(j,:)';
    u = u';
    
    u = sigmf(u, [1 0]);
    
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

Dteste = teste(:,6:8);
Xteste = teste(:,1:5);
erroTeste = 0;
YTeste = zeros(30,3);
for j = 1:30
    u = Xteste(j,:)*w;
    
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
    error(j,:) = Dteste(j,:)-YTeste(j,:);
    e = sum(abs(error(j,:)));
    if(e ~= 0)
        erroTeste = erroTeste + 1;
    end 
end