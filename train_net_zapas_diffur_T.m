%скрипт для диффур на основе нейросети для запаса 
clear
% https://docs.exponenta.ru/deeplearning/ug/solve-odes-using-a-neural-network.html#SolveODEsUsingANeuralNetworkExample-7% 	
%%
load zapas_formula_grunti % формирование массива
% t_f_g(:,[2:5,8:24,26,30:35,38,42,43])=[];% остались только те, что в формуле,1 - Твоздуха, 6 - Рвых Шакяй=Рвх_Шакяй_Красн, 7 - Твых Шакяй = Твх Шакяй Красн, 
% 25 - Рвых Шакяй Красн, 27 - Рвх Красн-154, 28 - Твх Красн-154, 29 - Рвых_Красн - 154, 31 - Рвых Шакяй Красн, 33 - Рвх Красн-140, 34 - Твх Красн-140, 35 - Рвых_Красн - 140, 
% 36,37 - плотности, 39,40 - молярные массы, 41 - запас на ВК, 42 - запас на ВК1, 43 - сумма запаса ВК+ВК1, 44 Тгрунта справ, 45 Тгрунта метео, 46 Тгрунта из данных
%% начальные и конечные давления по ниткам
P_nach_1_1 = t_f_g(:,6); % Рн, начальное абсолютное давление газа на участке, МПа; Шакяй - Краснознаменская 1
P_kon_1_1 = t_f_g(:,25);% Рк конечное абсолютное давлениея газа на участке, МПа; Шакяй - Краснознаменская 1
%% Расчет среднего давления на участках
P_av_1_1 = 2/3*(P_nach_1_1 + (P_kon_1_1.*P_kon_1_1)./(P_nach_1_1 + P_kon_1_1)); % Среднее давление на участке Шакяй - Краснознаменская 1
%% начальные и конечные температуры по ниткам и Тгрунта
T_nach_1_1 = t_f_g(:,7); % Тн Шакяй - Краснознаменская
T_kon_1_1 = 2.6; % Тк конечная температура газа, здесь взята из предоставленных данных, как допущение, Шакяй - Краснознаменская 
T_gr_meteo = t_f_g(:,45); % расчетное значние грунта (метео) 
%% Расчет средней температуры на участках. Простая формула
T_av_1_1_meteo = T_gr_meteo + (T_nach_1_1 - T_kon_1_1) ./ log(abs(T_nach_1_1 - T_gr_meteo)./abs(T_kon_1_1 - T_gr_meteo)); 
%% константы 
V_1 = 5.534; %* 1000;% геометрический объем трубы Шакяй - Краснознаменская 1 м3
% V_2 = 28.67; %* 1000;% геометрический объем трубы Краснознаменская - 154 м3
% V_3 = 10.974; %* 1000;% геометрический объем трубы Шакяй - Краснознаменская 2 м3
% V_4 = 25.633; %* 1000;% геометрический объем трубы 140 м3
%% псевдокритические давление и температура
M_x_a = 28.0135; % молярная доля азота кг/моль, определяемая по ГОСТ 30319.1
M_x_y = 44.010;   % молярная доля диоксида углерода кг/моль, определяемая по ГОСТ 30319.1 
density_standart_Shak = t_f_g(:,36); % плотность природного газа по воздуху измеренная Шаакяй
x_a = t_f_g(:,39);% концентрация азота, доли ед. измерения
x_y = t_f_g(:,40);% концентрация диоксида углерода, доли ед. измерения;
P_pk_1 = 1.808 *(26.831 - density_standart_Shak); % псеводкритическое давление на линии Шакяй, из методики 1999
T_pk_1 = 155.204*(0.564 + density_standart_Shak); % псеводкритическая температура на линии Шакяй, из методики 1999
%% приведенное давление 
P_pr_1_1 = P_av_1_1./P_pk_1; % приведенное давление на участке Шакяй - Краснознаменская 1 по справочным значениям Тгрунта
%% приведенная температура по значениям Тгрунта из Метео
T_pr_1_1_meteo = T_av_1_1_meteo./T_pk_1; % приведенная температура на участке Шакяй - Краснознаменская 1 по значениям Тгрунта из Метео
%% коэффициент сжимаемости по методике 1999 по значениям Тгрунта из Метео
Tau_1_1_meteo = 1-1.68*T_pr_1_1_meteo + 0.78*T_pr_1_1_meteo.^2 + 0.0107*T_pr_1_1_meteo.^3 ; % тау на участке Шакяй - Краснознаменская 1
Z_1999_1_1_meteo = 1 - 0.0241*P_pr_1_1./Tau_1_1_meteo; % коэффициент сжимаемости на участке Шакяй - Краснознаменская 1
%% 
P_c = 0.101325;% МПа;давление при стандартных условиях по ГОСТ 2939
T_c = 293.15;% К; температура при стандартных условиях по ГОСТ 2939
%% Запас газа итог на участке 1_1 по значениям Тгрунта из Метео
Zapas_1_1_meteo = V_1 * P_av_1_1 * T_c./(T_av_1_1_meteo .* Z_1999_1_1_meteo * P_c); % Объём запаса газа на участке Шакяй-Краснозн 1
y_diffur = V_1 * P_av_1_1 * T_c./(T_av_1_1_meteo .* Z_1999_1_1_meteo * P_c);
X1 = P_av_1_1;
% X2 = T_av_1_1_meteo;
% X3 = Z_1999_1_1_meteo;
save Zapas_meteo_1_1 Zapas_1_1_meteo
%% собираем данные для нейросети Шакяй
t_f_g(:,[1:5,8:24,26:35,37:41,42,43,44,46])=[];% остались только те, что в формуле,1 - Твоздуха, 6 - Рвых Шакяй=Рвх_Шакяй_Красн, 7 - Твых Шакяй = Твх Шакяй Красн, 
% 25 - Рвых Шакяй Красн, 27 - Рвх Красн-154, 28 - Твх Красн-154, 29 - Рвых_Красн - 154, 31 - Рвых Шакяй Красн, 33 - Рвх Красн-140, 34 - Твх Красн-140, 35 - Рвых_Красн - 140, 
% 36,37 - плотности, 39,40 - молярные массы, 41 - запас на ВК, 42 - запас на ВК1, 43 - сумма запаса ВК+ВК1, 44 Тгрунта справ, 45 Тгрунта метео, 46 Тгрунта из данных
%% собираем данные для нейросети из рассчетных величин
t_f_g_1_1 = [P_av_1_1, T_av_1_1_meteo, Z_1999_1_1_meteo, Zapas_1_1_meteo];
%%
t_f_g_1_1_outl = t_f_g_1_1; % создаем вспомогательную переменную, чтобы потом посмотреть на графике выбросы
for i= 1:width(t_f_g_1_1_outl) 
    t_f_g_1_1_outl(:,i) = filloutliers(t_f_g_1_1_outl(:,i),"previous","movmean",50);
end
%% рисуем выбросы
% figure
% plot(t_f_g_1_1(:,1)-t_f_g_1_1_outl(:,1));
% title("");
% %%
% figure
% plot(t_f_g_1_1(:,2)-t_f_g_1_1_outl(:,2));
% title("");
% %%
% figure
% plot(t_f_g_1_1(:,3)-t_f_g_1_1_outl(:,3));
% title("");
% %%
% figure
% plot(t_f_g_1_1(:,4)-t_f_g_1_1_outl(:,3));
% title("");
% %%
% figure
% plot(t_f_g_1_1_outl);
%%
t_f_g_1_1 = t_f_g_1_1_outl; % снова присваиваем массив, в котором теперь удалены выбросы
%% нормируем данные, считаем матожидание и ско
for i = 1:width(t_f_g_1_1) % 
    m(i) = mean(t_f_g_1_1(1:end-1,i));
    s(i) = std(t_f_g_1_1(1:end-1,i));
end
%% нормируем данные
for i= 1:width(t_f_g_1_1)
    t_f_g_1_1(:,i) = (t_f_g_1_1(:,i)- m(i))./s(i);
end
%% проверка нормирования, m1 = 0, s1 = 1
for i= 1:width(t_f_g_1_1) % 
    m1(i) = mean(t_f_g_1_1(1:end-1,i));
    s1(i) = std(t_f_g_1_1(1:end-1,i));
end
%%
x_train = t_f_g_1_1(1:3500,2);% обучающая выборка для входного параметра, столбец означает номер параметра 1 - Р, 2 - Т, 3 - Z,4 - запас
% figure
% plot(x_train);
% title("x train");
%% рисуем каждую входную переменную отдельно
% plot(x_train(:, 1));
% title("");
x_test = t_f_g_1_1(3501:end,2);% тестовая выборка
%% формируем строку ответов, это запас 
y = t_f_g_1_1(:,4);
y_train = y(1:3500); % обучающая выборка для выходного параметра
y_test = y(3501:end);% тестовая выборка для выходного параметра
y_test = - y_test + 0.5;
% figure
% plot(y_train);
% title("y train");
%% подали выходную переменную саму на себя, сдвинув на 1 шаг
% t_f_g_1_1(1:end-1 , 4) = t_f_g_1_1(2:end,4); 
%%
% x_train = x_train';
% y_train = y_train';
%%
x_test = x_test';
y_test = y_test';
%%
x = x_train;
%%
inputSize = 1;
layers = [
    featureInputLayer(inputSize,Normalization="none")
    fullyConnectedLayer(10)
    sigmoidLayer
    fullyConnectedLayer(1)
    sigmoidLayer];
%%
dlnet = dlnetwork(layers);
numEpochs = 15;
miniBatchSize = 100;
initialLearnRate = 0.5;
learnRateDropFactor = 0.5;
learnRateDropPeriod = 5;
momentum = 0.9;
icCoeff = 7;
%%
ads = arrayDatastore(x,IterationDimension=1);
mbq = minibatchqueue(ads,MiniBatchSize=miniBatchSize,MiniBatchFormat="BC");
%%
figure
set(gca,YScale="log")
lineLossTrain = animatedline(Color=[0.85 0.325 0.098]);
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss (log scale)")
grid on
%%
velocity = [];
iteration = 0;
learnRate = initialLearnRate;
start = tic;
%% Loop over epochs.
%Обучите сеть с помощью пользовательского учебного цикла. В течение каждой эпохи переставьте данные и цикл по мини-пакетам данных. Для каждого мини-пакета:
% Оцените градиенты модели и потерю с помощью dlfeval и modelGradients функции.
% Обновите сетевые параметры с помощью sgdmupdate функция.
% Отобразите прогресс обучения.
% Каждый learnRateDropPeriod эпохи, умножьте скорость обучения на learnRateDropFactor.
for epoch = 1:numEpochs

    % Shuffle data.
    mbq.shuffle

    % Loop over mini-batches.
    while hasdata(mbq)

        iteration = iteration + 1;

        % Read mini-batch of data.
        dlX = next(mbq);

        % Evaluate the model gradients and loss using dlfeval and the modelGradients function.
        [gradients,loss] = dlfeval(@modelGradients, dlnet, dlX, icCoeff);

        % Update network parameters using the SGDM optimizer.
        [dlnet,velocity] = sgdmupdate(dlnet,gradients,velocity,learnRate,momentum);

        % To plot, convert the loss to double.
        loss = double(gather(extractdata(loss)));
        
        % Display the training progress.
        D = duration(0,0,toc(start),Format="mm:ss.SS");
        addpoints(lineLossTrain,iteration,loss)
        title("Epoch: " + epoch + " of " + numEpochs + ", Elapsed: " + string(D))
        drawnow

    end
    % Reduce the learning rate.
    if mod(epoch,learnRateDropPeriod)==0
        learnRate = learnRate*learnRateDropFactor;
    end
end
%%
xTest = x_test';
% xTest = linspace(0,2,1000)';
adsTest = arrayDatastore(xTest,IterationDimension=1);
mbqTest = minibatchqueue(adsTest,MiniBatchSize=100,MiniBatchFormat="BC");
%%
yModel = modelPredictions(dlnet,mbqTest);
% yAnalytic = exp(xTest.^2);
V_1 = 5.534;
yAnalytic = 0.9*xTest+1.25 ;
%%
figure;
plot(xTest,yAnalytic,"*", LineWidth = 0.1)
hold on
plot(xTest,yModel,"--", LineWidth = 2)
legend("Analytic","Model")
xlabel("x")
ylabel("Запас")
% set(gca,YScale="log")
%%
difference = yAnalytic-yModel;
rmse = sqrt(mean((difference).^2)); % Calculate the root-mean-square error (RMSE).
difference1 = yModel -y_test';
rmse1 = sqrt(mean((difference1).^2));
difference2 = yAnalytic-y_test';
rmse2 = sqrt(mean((difference2).^2));
%%
figure;
subplot(2,1,1);
plot(y_test', LineWidth = 1);
title("Запас")
hold on;
plot(yModel,  "*", LineWidth = 0.01);
title("Запас мод")
hold on;
plot(yAnalytic, LineWidth = 1);
legend(["Запас тест" "Запас мод" "Запас аналит"])
xlabel("Time")
ylabel("Запас")
title("Запас тест Запас мод Запас аналит");
subplot(2,1,2)
plot(difference1, "*", LineWidth = 0.01);
hold on;
plot(difference2, "--",LineWidth = 1);
hold on;
plot(yAnalytic-yModel, LineWidth = 1);
legend(["RMSE Запас мод - Запас тест" "RMSE Запас аналит - Запас тест" "RMSE Запас аналит - Запас мод" ]);
xlabel("Time")
ylabel("RMSE")
title(["RMSE Запас мод - Запас тест = " + rmse1, "RMSE Запас аналит - Запас тест = " + rmse2, "RMSE Запас аналит - Запас мод = " + rmse]);
%% Функция градиентов модели
function [gradients,loss] = modelGradients(dlnet, dlX, icCoeff)
y = forward(dlnet,dlX);
% V_1 = 5.534;
% T_c = 293.15;
% Evaluate the gradient of y with respect to x. 
% Since another derivative will be taken, set EnableHigherDerivatives to true.
dy = dlgradient(sum(y,"all"),dlX,EnableHigherDerivatives=true);

% Define ODE loss.
% eq = dy + 2*y.*dlX;
eq = dy + 2*y.*dlX;
% y (x) = exp(P_av_1_1 .^ V_1 * T_c);
% Define initial condition loss.
ic = forward(dlnet,dlarray(0,"CB")) - 1;

% Specify the loss as a weighted sum of the ODE loss and the initial condition loss.
loss = mean(eq.^2,"all") + icCoeff * ic.^2;

% Evaluate model gradients.
gradients = dlgradient(loss, dlnet.Learnables);

end
%% Функция предсказаний модели
function Y = modelPredictions(dlnet,mbq)

Y = [];

while hasdata(mbq)

    % Read mini-batch of data.
    dlXTest = next(mbq);
    
    % Predict output using trained network.
    dlY = predict(dlnet,dlXTest);
    YPred = gather(extractdata(dlY));
    Y = [Y; YPred'];

end

end