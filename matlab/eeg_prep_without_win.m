for i = 101:108
    % Construct the filename dynamically
    filename = sprintf('%dFS2.mat', i); % Assuming files are named f10.mat, f11.mat, etc.

    % Load the file
    data = load(filename); % Load the .mat file into a struct

    % Access the content of the file (assuming the file has one main variable)
    fieldnames_data = fieldnames(data); % Get the field names in the struct
    content = data.(fieldnames_data{1}); % Access the first variable in the file

    % Assign to dynamically named variable (sig1, sig2, etc.)
    assignin('base', sprintf('sig%d', i - 100), content); % Subtract 9 to start sig1 for f10
end

%NO 109, 110

for i = 111:128
    % Construct the filename dynamically
    filename = sprintf('%dFS2.mat', i); % Assuming files are named f10.mat, f11.mat, etc.

    % Load the file  
    data = load(filename); % Load the .mat file into a struct

    % Access the content of the file (assuming the file has one main variable)
    fieldnames_data = fieldnames(data); % Get the field names in the struct
    content = data.(fieldnames_data{1}); % Access the first variable in the file

    % Assign to dynamically named variable (sig1, sig2, etc.)
    assignin('base', sprintf('sig%d', i - 102), content); % Subtract 9 to start sig1 for f10
end
 
%NO 129, 130

for i = 131:180
    % Construct the filename dynamically
    filename = sprintf('%dFS2.mat', i); % Assuming files are named f10.mat, f11.mat, etc.

    % Load the file
    data = load(filename); % Load the .mat file into a struct

    % Access the content of the file (assuming the file has one main variable)
    fieldnames_data = fieldnames(data); % Get the field names in the struct
    content = data.(fieldnames_data{1}); % Access the first variable in the file

    % Assign to dynamically named variable (sig1, sig2, etc.)
    assignin('base', sprintf('sig%d', i - 104), content); % Subtract 9 to start sig1 for f10
end

%NO 182

%WRONG TIMING IN 184

%NO 186

%NO 188

for i = 189:200
    % Construct the filename dynamically
    filename = sprintf('%dFS2.mat', i); % Assuming files are named f10.mat, f11.mat, etc.

    % Load the file
    data = load(filename); % Load the .mat file into a struct

    % Access the content of the file (assuming the file has one main variable)
    fieldnames_data = fieldnames(data); % Get the field names in the struct
    content = data.(fieldnames_data{1}); % Access the first variable in the file

    % Assign to dynamically named variable (sig1, sig2, etc.)
    assignin('base', sprintf('sig%d', i - 112), content); % Subtract 9 to start sig1 for f10
end

%---------------------------------

min_rows = inf;

for i = 1:88
    array = eval(sprintf('sig%d', i));
    [rows, cols] = size(array); 
    min_rows = min(min_rows, size(array(:,1))); % Update the smallest number of rows
end

% Display the results
disp(['Smallest number of rows: ', num2str(min_rows)]);

%sig78 WAS TERMINATED BEFORE 4 MINUTES RESULTING IN HAVING 105K POINTS
%INSTEAD OF 120K

%DECIDED TO ELIMINATE SIGNALS 183 AND 184

sigs = cell(1, 88);

% Fill the cell array with values in a loop
for i = 1:88
    var_name = sprintf('sig%d', i);
    
    % Access the variable from the workspace and assign it to the cell array
    if evalin('base', sprintf('exist(''%s'', ''var'')', var_name)) % Check if variable exists
        sigs{i} = evalin('base', var_name); % Access the variable
    else
        warning('Variable %s does not exist.', var_name);
    end
end

min_rows = min(cellfun(@(x) size(x, 1), sigs)); % Smallest row count
disp(['Smallest number of rows 222: ', num2str(min_rows)]);

for i = 1:88
    sigs{i} = sigs{i}(1:min_rows, :); % Change each array to 119520 length
end

%---------------------
%ARTIFACT DETECTION

fs = 500; % Sampling frequency (Hz)

% Artifact detection parameters
amplitude_threshold = 30000; % µV (example threshold for EEG)
gradient_threshold = 75; % Threshold for sharp transitions

% Initialize cell arrays for clean signals and artifact indices
clean_signals = cell(1, 88);
num_ar = zeros(88, 17); % 88 rows, 17 columns
num_ar_diff = zeros(88, 17); % 88 rows, 17 columns

for i = 1:88
    for j = 1:8
        signal = sigs{i}(:, j); % Extract the current signal
    
        % 1. Amplitude-based artifact detection
        high_amplitude_idx = find(abs(signal) > amplitude_threshold); % Get valid indices
        num_artifacts = length(high_amplitude_idx); % Number of artifacts
        num_ar(i, j) = num_artifacts;
    
        % 2. Gradient-based artifact detection
        gradient_signal = diff(signal); % First derivative
        high_gradient_idx = find(abs(gradient_signal) > gradient_threshold); % Get valid indices
        num_ar(i, j + 9) = length(high_gradient_idx);
 
        % Combine artifact indices
        artifact_idx = unique([high_amplitude_idx; high_gradient_idx]); % Combine unique indices

        % Optional: Remove or mark artifacts
        clean_signal(:, j) = signal;
        for idx = artifact_idx
            % Check for out-of-bounds indices
            if idx > 1 && idx < length(signal)
                % Replace with the average of the neighbors
                clean_signal(idx, j) = mean([clean_signal(idx - 1, j), clean_signal(idx + 1, j)]);
            elseif idx == 1
                % Handle the first element
                clean_signal(idx, j) = clean_signal(idx + 1, j);
            elseif idx == length(signal)
                % Handle the last element
                clean_signal(idx, j) = clean_signal(idx - 1, j);
            end
        end
    end
    clean_signals{i} = clean_signal; % Store the cleaned signal
end

% Display results
disp('Artifact detection complete!');

%------------------------
%BAND SEPARATION

bands = {
    'Delta', [0.5 4];
    'Theta', [4 8];
    'Alpha', [8 13];
    'Beta', [13 30];
    'Gamma', [30 100];
};

t = 1:min_rows
% Preallocate storage for filtered signals
filtered_signals = cell(88*5, 9);
r = 0

for k = 1:88 
    for j = 1:8
        for i = 1:size(bands, 1)
            band_name = bands{i, 1};
            freq_range = bands{i, 2};
            
    
            % Design a bandpass Butterworth filter
            [b, a] = butter(4, freq_range / (fs / 2), 'bandpass'); % Normalize by Nyquist frequency
    
            % Apply the filter
            filtered_signals{i + r, 1} = bands{i, 1};
            filtered_signals{i + r, j+1} = filtfilt(b, a, clean_signals{k}(:, j));
        end
    end
    r = r + 5;
end

%----------------------------
%SEPARATION BY 2 SECOND WINDOWS

window_size = 1000;

% Initialize output cell array
win_sigs = cell(120*440, 10);

q = 0;
% Process each signal in the cell array
for i = 1:440
    for j = 2:9
        signal = filtered_signals{i, j}; % Extract the current signal
        [total_points, num_channels] = size(signal);
    
        % Calculate the number of windows
        num_windows = floor(total_points / window_size) + 1;

        for w = 1:num_windows
            start_idx = (w - 1) * window_size + 1;
            end_idx = w * window_size;
            if w == 120
                end_idx = w * window_size - 500;
            end
            win_sigs{w + q*num_windows, j + 1} = signal(start_idx:end_idx, :); % Section the signal
            win_sigs{w + q*num_windows, 2} = filtered_signals{i, 1};
            win_sigs{w + q*num_windows, 1} = " part " + string(w);
        end
    end
    q = q + 1
end

%----------------------------
%CORRELATION COMPUTATION

sig_corr = cell(44*5*num_windows, 9)
p = 0

for k = 1:44*5*num_windows
    c = k + p*5*num_windows;
    for j = 3:10
        sig_corr{k, j} = corr(win_sigs{c, j}, win_sigs{c + 5*num_windows, j});
    end
    if mod(c, 5*num_windows) == 0
        p = p + 1
    end
end

for k = 1:44*5*num_windows
    sig_corr{k, 2} = win_sigs{k, 2};
    sig_corr{k, 1} = "pair " + string(floor(k / (5*num_windows) + 1)) + win_sigs{k, 1};
end


    
writecell(sig_corr, 'correlations_array120.csv');

% Check the current folder for 'output.csv'
disp('Cell array written to correlations_array120.csv');