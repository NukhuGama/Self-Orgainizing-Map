clc;
clear all;
% Membaca data di exel
dataSet = csvread('Tugas 2 ML Genap 2018-2019 Dataset Tanpa Label.xls'); 

train_data = dataSet(:,1:2);
[dataRow, dataCol] = size(train_data);

% Menghitung banyaknya baris dan kolom dalam SOM
% Dimensi (Baris x Kolom)
somRow = 25 %input('Masukan Dimensi Baris :');
somCol = 25 %input('Masukan Dimensi Kolom :');

% Banyaknya Iterasi untuk convergence 
Iteration = 150 %input('Masukan banyaknya Iterasi :');

% Menentukan label sendiri untuk membuat cluster dengan membandingkan data
% kolom 1 dengan data kolom 2
for n=1:dataRow
    b(n) = train_data(n,1)/train_data(n,2);
    % Ambil ceilnya
    b(n) = ceil(b(n))
    
end;
% Ubah ke matriks transpost
y = transpose(b)

% y = dataSet(:,2)

%%=========== Parameter untuk Setting SOM ===================================
% ukuran lebar untuk winning neuron dalam topological neighbourhood 
width_Initial = 15;

% waktu konstan untuk topologi neighbourhood 
t_width = Iteration/log(width_Initial);

% waktu awal learning rate
learningRate_Initial = 1;

% waktu konstan untuk learning rate
t_learningRate = Iteration;

fprintf('\nAmbil bobot Neuron secara Acak ...\n')
% Initial bobot(weight) vector untuk neuron
somMap = randInitializeWeights(somRow,somCol,dataCol);

% Plot  training data
plotData(train_data, y);
hold on;

for t = 1:Iteration
    
    % Ukuran topological neighbourhood untuk winning neuron dalam iterasi t 
    width = width_Initial*exp(-t/t_width);
    
    width_Variance = width^2;
    
    % Laju learning rate pada iterasi t pada waktu tertentu
    learningRate = learningRate_Initial*exp(-t/t_learningRate);
    
    
    % Mencegah learning rate menjadi terkecil
    if learningRate <0.025
        learningRate = 0.1;
    end
   
    % Menghitung Euclidean distance antara setiap neuron dan input
    [euclideanDist, index] = findBestMatch( train_data, somMap, somRow, ...
                                            somCol, dataRow, dataCol );
    % indeks untuk winning neuron 
    [minM,ind] = min(euclideanDist(:)); 
    [win_Row,win_Col] = ind2sub(size(euclideanDist),ind);
    
    % Menghitung fungsi neighborhood pada tiap neuron
    neighborhood = computeNeighbourhood( somRow, somCol, win_Row, ...
                                            win_Col, width_Variance);
    
    % Update  bobot untuk setiap neuron di grid
    somMap = updateWeight( train_data, somMap, somRow, somCol, ...
                            dataCol, index, learningRate, neighborhood)
  
    % Bobot vector neuron 
    dot = zeros(somRow*somCol, dataCol);
    % Matrix untuk SOM plot grid
    matrix = zeros(somRow*somCol,1);
    % Matrix Penghapusan untuk SOM plot grid  
    matrix_old = zeros(somRow*somCol,1);
    
   
    ind = 1;  
    hold on;
    f1 = figure(1);
    set(f1,'name',strcat('Iteration #',num2str(t)),'numbertitle','off');

    % Ambil bobot(weigth) dari neuron tiap vector
    for r = 1:somRow
        for c = 1:somCol      
            dot(ind,:)=reshape(somMap(r,c,:),1,dataCol);
            ind = ind + 1;
        end
    end

    % Plot SOM
    for r = 1:somRow
        Row_1 = 1+somRow*(r-1);
        Row_2 = r*somRow;
        Col_1 = somRow*somCol;

        matrix(2*r-1,1) = plot(dot(Row_1:Row_2,1),dot(Row_1:Row_2,2),'--ro','LineWidth',0.5,'MarkerEdgeColor','g','MarkerFaceColor','g','MarkerSize',2);
        matrix(2*r,1) = plot(dot(r:somCol:Col_1,1),dot(r:somCol:Col_1,2),'--ro','LineWidth',0.5,'MarkerEdgeColor','g','MarkerFaceColor','g','MarkerSize',2);

        matrix_old(2*r-1,1) = matrix(2*r-1,1);
        matrix_old(2*r,1) = matrix(2*r,1);

    end

    % Menghapus plot SOM dari iterasi sebelumnya
    if t~=Iteration  
        for r = 1:somRow
            delete(matrix_old(2*r-1,1));
            delete(matrix_old(2*r,1));
            drawnow;
        end
    end  
    
end
legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','location','eastoutside')


function neighbourhood_Function = computeNeighbourhood( somRow, ... 
                                  somCol, win_Row, win_Col, width_Variance)
% Fungsi ini akan menghitung jarak lateral(lateral distance) antara neurons i dan winning neurons 

    % Inisialisasi matrix untuk menyimpan euclidean distance antara tiap
    % neuron dengan winning neuron untuk menghitung fungsi neighbour
    neighbourhood_Function = zeros(somRow, somCol);
    
    for r = 1:somRow
       for c = 1:somCol
           if (r == win_Row) && (c == win_Col)
               % Fungsi neihbour untuk winning neuron
               neighbourhood_Function(r,c) = 1;
           else
               % fungsi neighbour untuk neuron lain 
               distance = (win_Row - r)^2+(win_Col - c)^2;
               neighbourhood_Function(r,c) = exp(-distance/(2*width_Variance));
           end    
       end
    end
end

function weight_Vector = randInitializeWeights( row, column, dataCol )
%Inisialisai bobot(weight) vector pada tiap neuron secara acak antara 0 dan 1

% Menginisialisasi bobot vector matrix
weight_Vector = zeros(row, column, dataCol); 

for r = 1:row
    for c = 1:column
        weight_Vector(r,c,:) = rand(1,dataCol);
    end
end

end

function [euclidean_Distance, i] = findBestMatch( train_data, somMap, somRow,...
                                somCol, dataRow, dataCol )
% Fungsi ini akan mencari best matched vector(winning neuron) sesuai dengan input image
% Inisialisasi matrix untuk menyimpan Euclidean distance antara input vector dan tiap neuron
    euclidean_Distance = zeros(somRow, somCol);

    i = randi([1 dataRow]);
    
    for r = 1:somRow
        for c = 1:somCol
            V = train_data(i,:) - reshape(somMap(r,c,:),1,dataCol);
            euclidean_Distance(r,c) = sqrt(V*V');
        end
    end

end

function plotData( train_data, y )
% Fungsi plot data 
    figure; 
    hold on;
    cluster_1 = find(y==1);
    cluster_2 = find(y==2);
    cluster_3 = find(y==3);
    cluster_4 = find(y==4);
    
    
    plot(train_data(cluster_1,1),train_data(cluster_1,2),'y.','LineWidth',2,'MarkerSize',20);
    plot(train_data(cluster_2,1),train_data(cluster_2,2),'r.','LineWidth',2,'MarkerSize',20);
    plot(train_data(cluster_3,1),train_data(cluster_3,2),'c.','LineWidth',2,'MarkerSize',20);
    plot(train_data(cluster_4,1),train_data(cluster_4,2),'b.','LineWidth',2,'MarkerSize',20);

    set(gcf,'un','n','pos',[0,0,1,1]);figure(gcf)

    %hold on;

%     legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','location','eastoutside')
    hold off;


end

function WeightVectorUpdated = updateWeight(  train_data, somMap, somRow, ... 
                        somCol, dataCol, index, learningRate, neighborhood)
% Fungsi ini  akan mengupdate semua neuron tergantung jarak antara winning neuron dan neuron lain  

    WeightVectorUpdated = zeros(somRow, somCol, dataCol);
    
    for r = 1: somRow
       for c = 1:somCol
           
           % Reshape dimensi bobot vector aktual 
           currentWeightVector = reshape(somMap(r,c,:),1,dataCol);
          
           % Update bobot(weight) vector pada tiap neuron 
           WeightVectorUpdated(r,c,:) = currentWeightVector + learningRate*neighborhood(r,c)*(train_data(index,:)-currentWeightVector);
            
       end
    end 
end
