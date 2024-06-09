%% This script reads a given dataset and generates a new dataset
%% (named "measurements_dataset_nn.txt") with the predictions of
%% the optimum configuration for the RIS using the NN

%% DEPRECATED PARAMETERS
PREV_TS_WINDOW = 0; % DEPRECATED -> set to false
GLOBAL_SHUFFLE = false; % DEPRECATED -> set to false

%% GENERAL PARAMS
N_EXECUTIONS = 2; % CAN BE CHANGED -> 10 is a nice value

%% FOR-LOOP PARAMS
NN_LAYERS_ARRAY = 1; % TODO: Not used yet
RIS_SIZE_ARRAY = [4, 16, 64]; % 4, 16, 64
N_DIFFERENT_CHANNELS_ARRAY = 10:10:100; % [10, 100, 1000]; 100;
N_MEASUREMENTS_ARRAY = 1000; % 10:10:100
VAR_CEE_ARRAY = 0.5; % 0.1:0.1:1.0; % 0.5;
NN_SIZE_SWEEP = 200; %40:40:200; % Number of neurons
LR_SWEEP = 0.01; % [0.0001, 0.001, 0.01, 0.1, 1]; % 0.01; % Learning rates
EPOCHS_SWEEP = 10000; % Maximum number of epochs
TRAINING_ALGORITHM_SWEEP = "trainscg"; % "min_distance_channels_without_NN"; % "min_distance_configuration_without_NN"; % "trainscg"; % ["trainscg", "trainlm", "traingdm"]; % Training algorithm

global_start_time = tic;

%% Save results to file
results_filename = ("results_file_" + strrep(datestr(datetime("now")), ":", "_") + ".txt");
fileID = fopen(results_filename,'w');
fprintf(fileID, "# RIS_size    N_different_channels    N_measurements    Var_CEE    NN_layers    NN_neurons    LR(useless so far)    N_epochs_max    training_algorithm    Training_time (s)    N_epochs_used    training_MSE_NN    training_MSE_direct    validation_MSE_NN    validation_MSE_direct    test_MSE_NN    test_MSE_direct\n");
fclose(fileID);

for RIS_size = RIS_SIZE_ARRAY
    for n_different_channels = N_DIFFERENT_CHANNELS_ARRAY
        for n_measurements = N_MEASUREMENTS_ARRAY
            for var_CEE = VAR_CEE_ARRAY
                Std_CEE=sqrt(var_CEE);

                %% DEBUG TO LIMIT DATASET TO N_entries entries
                M = readtable("measurements_dataset_RIS" + RIS_size + "_1000.txt");

                M = M{:,:};
                M = M(1:n_different_channels, :);

                fileID2 = fopen("measurements_dataset_tmp.txt", "w");

                for rep = 1:n_measurements
                    % Shuffle
                    M = M(randperm(size(M, 1)), :);

                    angle_g_real_array = M(:, 1:RIS_size);
                    angle_h_real_array = M(:, (RIS_size + 1):(RIS_size * 2));
                    phaseShifts_RIS_optimum_array = M(:, (RIS_size * 4 + 1):(RIS_size * 5));

                    for i = 1:size(M, 1)
                        angle_g_estimated_array = mod(angle(exp(1j * angle_g_real_array(i, :)) + Std_CEE*randn(size(angle_g_real_array(i, :))) + 1j * Std_CEE*randn(size(angle_g_real_array(i, :)))), 2 * pi);
                        angle_h_estimated_array = mod(angle(exp(1j * angle_h_real_array(i, :)) + Std_CEE*randn(size(angle_h_real_array(i, :))) + 1j * Std_CEE*randn(size(angle_h_real_array(i, :)))), 2 * pi);
                        fprintf(fileID2, join(repmat("%f",1, (5 * RIS_size)),', ') + "\n", angle_g_real_array(i, :).', angle_h_real_array(i, :).', angle_g_estimated_array.', angle_h_estimated_array.', phaseShifts_RIS_optimum_array(i, :).');
                    end
                end
                fclose(fileID2);

                %% THE DATASET HAS BEEN GENERATED
                for NN_layers = NN_LAYERS_ARRAY
                    for nn_size = NN_SIZE_SWEEP
                        for training_algorithm = TRAINING_ALGORITHM_SWEEP
                            for lr = LR_SWEEP
                                for epochs = EPOCHS_SWEEP
                                    for execution = 1:N_EXECUTIONS
                                        %% Load input dataset
                                        M = readtable('measurements_dataset_tmp.txt');
                                        M = M{:,:};

                                        M_orig = M;
                                        M = [M_orig((1 + PREV_TS_WINDOW):end, :), zeros(size(M_orig, 1) - PREV_TS_WINDOW, RIS_size * 2 * PREV_TS_WINDOW)];

                                        %% Concat previous time steps
                                        for tshift = 0:(PREV_TS_WINDOW - 1)
                                            pos_start = size(M_orig, 2) + 1 + tshift * (2 * RIS_size);
                                            pos_end = pos_start + (2 * RIS_size) - 1;
                                            % Noisy values of prev samples as input
                                            M(:, pos_start : pos_end) = M_orig(tshift + (1:size(M)), (RIS_size * 2 + 1):(RIS_size * 4));
                                        end

                                        angle_g_real_array = M(:, 1:RIS_size);
                                        angle_h_real_array = M(:, (RIS_size + 1):(RIS_size * 2));
                                        angle_g_estimated_array = M(:, (RIS_size * 2 + 1):(RIS_size * 3));
                                        angle_h_estimated_array = M(:, (RIS_size * 3 + 1):(RIS_size * 4));
                                        phaseShifts_RIS_optimum_array = M(:, (RIS_size * 4 + 1):(RIS_size * 5));

                                        input = M(:, [(RIS_size * 2 + 1):(RIS_size * 4), (RIS_size * 5 + 1):size(M, 2)]);

                                        labels_channels = [angle_g_real_array angle_h_real_array];

                                        input = input.';
                                        output = phaseShifts_RIS_optimum_array;
                                        output = output.';

                                        x = input;
                                        t = output;

                                        %% Encoding for x and t: https://stats.stackexchange.com/questions/218407/encoding-angle-data-for-neural-network
                                        x = [sin(x); cos(x)];
                                        t = [sin(t); cos(t)];
                                        
                                        TRAIN_RATIO = 0.7;
                                        VALIDATION_RATIO = 0.1;
                                        
                                        start_train = 1;
                                        end_train = round(size(x, 2) * TRAIN_RATIO);
                                        start_validation = end_train + 1;
                                        end_validation = round(start_validation + (size(x, 2) * VALIDATION_RATIO) - 1);
                                        start_test = end_validation + 1;
                                        end_test = size(x, 2);
                            
                                        x_train = x(:, start_train:end_train);
                                        x_validation = x(:, start_validation:end_validation);
                                        x_test = x(:, start_test:end_test);
                            
                                        t_train = t(:, start_train:end_train);
                                        t_validation = t(:, start_validation:end_validation);
                                        t_test = t(:, start_test:end_test);

                                        start_time = tic;
                                        if (training_algorithm == "min_distance_channels_without_NN")
                                            %% Minimum euclidean distance calculation without using an NN
                                            labels_train = labels_channels(start_train:end_train, :);

                                            labels = unique(labels_train, 'rows');

                                            labels = [sin(labels) cos(labels)].';

                                            y_train = zeros(size(labels, 1), size(x_train, 2));
                                            for val_i = 1:size(x_train, 2)
                                                val = x_train(:, val_i);
                                                best_label = -1 * ones(size(val));
                                                best_distance = -1;
                                                for label_i=1:size(labels, 2)
                                                    label = labels(:, label_i);
                                                    distance = sqrt(sum((val - label) .^ 2));
                                                    if ((best_distance == -1) || (distance < best_distance))
                                                        best_label = label;
                                                        best_distance = distance;
                                                    end
                                                end
                                                y_train(:, val_i) = best_label;
                                            end

                                            y_validation = zeros(size(labels, 1), size(x_validation, 2));
                                            for val_i = 1:size(x_validation, 2)
                                                val = x_validation(:, val_i);
                                                best_label = -1 * ones(size(val));
                                                best_distance = -1;
                                                for label_i=1:size(labels, 2)
                                                    label = labels(:, label_i);
                                                    distance = sqrt(sum((val - label) .^ 2));
                                                    if ((best_distance == -1) || (distance < best_distance))
                                                        best_label = label;
                                                        best_distance = distance;
                                                    end
                                                end
                                                y_validation(:, val_i) = best_label;
                                            end

                                            y_test = zeros(size(labels, 1), size(x_test, 2));
                                            for val_i = 1:size(x_test, 2)
                                                val = x_test(:, val_i);
                                                best_label = -1 * ones(size(val));
                                                best_distance = -1;
                                                for label_i=1:size(labels, 2)
                                                    label = labels(:, label_i);
                                                    distance = sqrt(sum((val - label) .^ 2));
                                                    if ((best_distance == -1) || (distance < best_distance))
                                                        best_label = label;
                                                        best_distance = distance;
                                                    end
                                                end
                                                y_test(:, val_i) = best_label;
                                            end

                                            y_train = mod(atan2(y_train(1:(2 * RIS_size), :), y_train((2* RIS_size + 1):(2 * RIS_size * 2), :)), 2 * pi);
                                            y_train = mod(2 * pi - y_train(1:RIS_size, :) - y_train((RIS_size + 1):(2 * RIS_size), :), 2 * pi);
                                            y_train = [sin(y_train); cos(y_train)];

                                            y_validation = mod(atan2(y_validation(1:(2 * RIS_size), :), y_validation((2* RIS_size + 1):(2 * RIS_size * 2), :)), 2 * pi);
                                            y_validation = mod(2 * pi - y_validation(1:RIS_size, :) - y_validation((RIS_size + 1):(2 * RIS_size), :), 2 * pi);
                                            y_validation = [sin(y_validation); cos(y_validation)];

                                            y_test = mod(atan2(y_test(1:(2 * RIS_size), :), y_test((2* RIS_size + 1):(2 * RIS_size * 2), :)), 2 * pi);
                                            y_test = mod(2 * pi - y_test(1:RIS_size, :) - y_test((RIS_size + 1):(2 * RIS_size), :), 2 * pi);
                                            y_test = [sin(y_test); cos(y_test)];

                                            elapsed_time = toc(start_time);
                                        else
                                            hidden_sizes_mat = repmat([nn_size], 1, NN_layers); %% TODO: Implement correctly
                                            net = feedforwardnet(hidden_sizes_mat, training_algorithm); % , 'trainscg' -> Scaled conjugate gradient descent
                                            net.trainParam.lr = lr;  % setting the learning rate
                                            net.trainParam.epochs = epochs;  % setting number of epochs
                                            net.trainParam.max_fail = 10;
                                            net.trainParam.showWindow = false;
                                            net.trainParam.showCommandLine = true;
        
                                            [net, tr] = train(net,x_train,t_train,'UseParallel','yes','UseGPU','yes'); % 'UseParallel','yes','UseGPU','yes'
        
                                            elapsed_time = toc(start_time);
        
                                            %view(net)
        
                                            y_train = net(x_train);
                                            y_validation = net(x_validation);
                                            y_test = net(x_test);
                                        end
                                        
                                        %% Decoding for x, t and y
                                        x_train = atan2(x_train(1:size(input, 1), :), x_train((size(input, 1) + 1):(size(input, 1) * 2), :));
                                        t_train = mod(atan2(t_train(1:RIS_size, :), t_train((RIS_size + 1):(RIS_size * 2), :)), 2 * pi);
                                        y_train = mod(atan2(y_train(1:RIS_size, :), y_train((RIS_size + 1):(RIS_size * 2), :)), 2 * pi);

                                        x_validation = atan2(x_validation(1:size(input, 1), :), x_validation((size(input, 1) + 1):(size(input, 1) * 2), :));
                                        t_validation = mod(atan2(t_validation(1:RIS_size, :), t_validation((RIS_size + 1):(RIS_size * 2), :)), 2 * pi);
                                        y_validation = mod(atan2(y_validation(1:RIS_size, :), y_validation((RIS_size + 1):(RIS_size * 2), :)), 2 * pi);

                                        x_test = atan2(x_test(1:size(input, 1), :), x_test((size(input, 1) + 1):(size(input, 1) * 2), :));
                                        t_test = mod(atan2(t_test(1:RIS_size, :), t_test((RIS_size + 1):(RIS_size * 2), :)), 2 * pi);
                                        y_test = mod(atan2(y_test(1:RIS_size, :), y_test((RIS_size + 1):(RIS_size * 2), :)), 2 * pi);


                                        %% Calculate direct estimations as baseline
                                        direct_estimation_train = mod(2 * pi - angle_g_estimated_array(start_train:end_train, :) - angle_h_estimated_array(start_train:end_train, :), 2 * pi);
                                        direct_estimation_validation = mod(2 * pi - angle_g_estimated_array(start_validation:end_validation, :) - angle_h_estimated_array(start_validation:end_validation, :), 2 * pi);
                                        direct_estimation_test = mod(2 * pi - angle_g_estimated_array(start_test:end_test, :) - angle_h_estimated_array(start_test:end_test, :), 2 * pi);

                                        mse_nn_train = mean((pi-abs(abs(t_train - y_train) - pi)).^2,'all');
                                        mse_direct_train = mean((pi-abs(abs(t_train - direct_estimation_train.') - pi)).^2,'all');

                                        mse_nn_validation = mean((pi-abs(abs(t_validation - y_validation) - pi)).^2,'all');
                                        mse_direct_validation = mean((pi-abs(abs(t_validation - direct_estimation_validation.') - pi)).^2,'all');

                                        mse_nn_test = mean((pi-abs(abs(t_test - y_test) - pi)).^2,'all');
                                        mse_direct_test = mean((pi-abs(abs(t_test - direct_estimation_test.') - pi)).^2,'all');

                                        fprintf("MSE for NN of size %i, LR = %f, epochs = %i:\n", nn_size, lr, epochs);
                                        fprintf("    Training time: %f seconds, epochs: %i\n", max(tr.time), tr.num_epochs);
                                        fprintf("    training: NN = %f, direct = %f\n", mse_nn_train, mse_direct_train);
                                        fprintf("    validation: NN = %f, direct = %f\n", mse_nn_validation, mse_direct_validation);
                                        fprintf("    test: NN = %f, direct = %f\n", mse_nn_test, mse_direct_test);

                                        %% SAVE RESULTS TO FILE FOR PROCESSING WITH TU_VIENNA
                                        x = x_test;
                                        t = t_test;
                                        y = y_test;

                                        direct_estimation = mod(2 * pi - angle_g_estimated_array(start_test:end_test, :) - angle_h_estimated_array(start_test:end_test, :), 2 * pi);
                                        angle_g_real_array = angle_g_real_array(start_test:end_test, :);
                                        angle_h_real_array = angle_h_real_array(start_test:end_test, :);

                                        %% Save results to file for processing with TU Vienna simulator
                                        fileID = fopen('measurements_dataset_nn.txt','w');
                                        for i = 1:size(x, 2)
                                            fprintf(fileID, join(repmat("%f",1, (6 * RIS_size)),', ') + "\n", mod(angle_g_real_array(i, :), 2 * pi).', mod(angle_h_real_array(i, :), 2 * pi).', mod(x(1:(RIS_size * 2), i), 2 * pi), mod(t(:, i), 2 * pi), mod(y(:, i), 2 * pi));
                                        end
                                        fclose(fileID);


                                        %% TU Vienna results
                                        throughput_random = -1;
                                        coded_BER_random = -1;
                                        uncoded_BER_random = -1;
                                        throughput_direct = -1;
                                        coded_BER_direct = -1;
                                        uncoded_BER_direct = -1;
                                        throughput_nn = -1;
                                        coded_BER_nn = -1;
                                        uncoded_BER_nn = -1;
                                        throughput_optimum = -1;
                                        coded_BER_optimum = -1;
                                        uncoded_BER_optimum = -1;

                                        %% Print all results to file
                                        fileID = fopen(results_filename,'a');
                                        fprintf(fileID, "%i %i %i %f %i %i %f %i %s %f %i %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", ...
                                            RIS_size, n_different_channels, n_measurements, var_CEE, NN_layers, nn_size, ...
                                            lr, epochs, training_algorithm, max(tr.time), tr.num_epochs, mse_nn_train, mse_direct_train, ...
                                            mse_nn_validation, mse_direct_validation, mse_nn_test, mse_direct_test, throughput_random, ...
                                            coded_BER_random, uncoded_BER_random, throughput_direct, coded_BER_direct, uncoded_BER_direct, ...
                                            throughput_nn, coded_BER_nn, uncoded_BER_nn, throughput_optimum, coded_BER_optimum, uncoded_BER_optimum);
                                        fclose(fileID);

                                    end % for N_EXECUTIONS
                                end % for EPOCHS
                            end % for LR
                        end % for TRAINING_ALGORITHM
                    end % for NN_SIZE
                end % for VAR_CEE
            end % for N_MEASUREMENTS
        end % for N_DIFFERENT_CHANNELS
    end % for RIS_SIZE
end % for NN_LAYERS

toc(global_start_time)
