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
rxBedford = rxsite("Name","Rectorate", ...
    "Latitude", 45.75366000336922, ...
    "Longitude", 21.225070006036553);
rxStA = rxsite("Name","Iulius Mall", ...
    "Latitude", 45.76587802049761, ...
    "Longitude", 21.22646131181263);
rxGPD = rxsite("Name","Student Campus", ...
    "Latitude", 45.748774167169216, ...
    "Longitude", 21.24115535510316);
rxs = [rxBedford, rxStA, rxGPD];
show(rxs);

% Achieve Line-of-Sight Link Visibility
% A challenge for 5G communication is achieving a successful link in the
% presence of terrain and other obstacles, since propagation losses
% increase at high frequency.
los(tx,rxs);

% Adjust antenna heights in order to achieve line-of-sight visibility.
% Place antennas on structures at receiver sites. Assume 6 m utility poles for Bedford
% and St. Anselm sites, and 15 m antenna pole at Goffstown Police Department.
rxBedford.AntennaHeight = 1;
rxStA.AntennaHeight = 1;
rxGPD.AntennaHeight = 1;

% Increase height of antenna at base station until line-of-sight
% is achieved with all receiver sites
tx.AntennaHeight = 1;
while ~all(los(tx,rxs))
    tx.AntennaHeight = tx.AntennaHeight + 1;
end

% Display line-of-sight
los(tx,rxs);
disp("Antenna height required for line-of-sight: " + tx.AntennaHeight + " m");
