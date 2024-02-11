% Create Base Station Site in 28 GHz Band
% Place it on top of the Coloseum in Rome
fq = 28e9; % 28 GHz
tx = txsite("Name","Colloseum Rome", ...
    "Latitude", 41.89073481346499, ...
    "Longitude", 12.492230899998866,...
    "TransmitterPower",1, ...
    "TransmitterFrequency",fq);
show(tx);

% Create Receiver Sites
% Create three receiver sites in the area and show the sites on the map
rx = rxsite("Name","Politechnic University Timisoara", ...
    "Latitude", 45.747546009667865, ...
    "Longitude", 21.226313580839275);
show(rx);

% Achieve Line-of-Sight Link Visibility
% A challenge for 5G communication is achieving a successful link in the
% presence of terrain and other obstacles, since propagation losses
% increase at high frequency.
los(tx, rx);

% Adjust antenna heights in order to achieve line-of-sight visibility.
rx.AntennaHeight = 1;

% Increase height of antenna at base station until line-of-sight
% is achieved with the receiver site in UPT
tx.AntennaHeight = 50000;
while ~all(los(tx, rx))
    tx.AntennaHeight = tx.AntennaHeight + 100;
end
% Tx antenna height require to reach LOS is around 55.900m

% Display line-of-sight
los(tx, rx);
disp("Antenna height required for line-of-sight: " + tx.AntennaHeight + " m");
