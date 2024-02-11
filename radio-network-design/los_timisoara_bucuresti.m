% Create Base Station Site in 28 GHz Band
% Place it on top of the Politehnica Timisoara University
fq = 28e9; % 28 GHz
tx = txsite("Name","Politechnic University Timisoara", ...
    "Latitude", 45.747546009667865, ...
    "Longitude", 21.226313580839275, ...
    "TransmitterPower",1, ...
    "TransmitterFrequency",fq);
show(tx);

% Create Receiver Sites
% Create three receiver sites in the area and show the sites on the map
rx = rxsite("Name","Politechnic University Bucharest", ...
    "Latitude", 44.43854474857753, ...
    "Longitude", 26.05149597759722);
show(rx);

% Achieve Line-of-Sight Link Visibility
% A challenge for 5G communication is achieving a successful link in the
% presence of terrain and other obstacles, since propagation losses
% increase at high frequency.
los(tx, rx);

% Adjust antenna heights in order to achieve line-of-sight visibility.
% Place antennas on structures at receiver sites. Assume 6 m utility poles for Bedford
% and St. Anselm sites, and 15 m antenna pole at Goffstown Police Department.
rx.AntennaHeight = 10000;

% Increase height of antenna at base station until line-of-sight
% is achieved with the receiver site in UPB
%tx.AntennaHeight = 5000;
%while ~all(los(tx, rx))
%    tx.AntennaHeight = tx.AntennaHeight + 100;
%end
% Tx antenna height need to be aroun 11.400m so that it can reach LOS

% Display line-of-sight
%los(tx, rx);
%disp("Tx Antenna height required for line-of-sight: " + tx.AntennaHeight + " m");
