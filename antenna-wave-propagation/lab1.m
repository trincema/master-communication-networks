% Map parameters
viewer = siteviewer;
viewer.Basemap = "openstreetmap";
viewer = siteviewer("Basemap","openstreetmap", "Buildings","manhattan.osm");

% Create Base Station Site in 28 GHz Band
% Place it on top of the Politehnica Timisoara University
fq = 28e9; % 28 GHz
tx = txsite("Name","Politechnic University Timisoara", ...
    "Latitude", 45.747546009667865, ...
    "Longitude", 21.226313580839275, ...
    "TransmitterPower", 1, ...
    "TransmitterFrequency", fq);
show(tx);

% Create Receiver Sites
% Create three receiver sites in the area and show the sites on the map
rx = rxsite("Name","Rectorate", ...
    "Latitude", 45.75366000336922, ...
    "Longitude", 21.225070006036553);
show(rxs);

% Achieve Line-of-Sight Link Visibility
% A challenge for 5G communication is achieving a successful link in the
% presence of terrain and other obstacles, since propagation losses
% increase at high frequency.
los(tx,rxs);


function element = reflectorCrossedDipoleElement(fq, showAntenna)
    %reflectorCrossedDipoleElement   Design reflector-backed crossed dipole antenna element
    if nargin < 2
        showAntenna = false;
    end
    
    lambda = physconst("lightspeed")/fq;
    offset = lambda/50;
    gndspacing = lambda/4;
    gndLength = lambda;
    gndWidth = lambda;
    
    % Design crossed dipole elements
    d1 = design(dipole,fq);
    d1.Tilt = [90,-45];
    d1.TiltAxis = ["y","z"];
    d2 = copy(d1);
    d2.Tilt = 45;
    d2.TiltAxis = "x";
    
    % Design reflector
    r = design(reflector,fq);
    r.Exciter = d1;
    r.GroundPlaneLength = gndLength;
    r.GroundPlaneWidth = gndWidth;
    r.Spacing = gndspacing;
    r.Tilt = 90;
    r.TiltAxis = "y";
    if showAntenna
        show(r)
    end
    
    % Form the crossed dipole backed by reflector
    refarray = conformalArray;
    refarray.ElementPosition(1,:) = [gndspacing 0 0];
    refarray.ElementPosition(2,:) = [gndspacing+offset 0 0];
    refarray.Element = {r, d2};
    refarray.Reference = "feed";
    refarray.PhaseShift = [0 90];
    if showAntenna
        show(refarray);
        view(65,20)
    end

    % Create custom antenna element from pattern
    [g,az,el] = pattern(refarray,fq);
    element = phased.CustomAntennaElement;
    element.AzimuthAngles = az;
    element.ElevationAngles = el;
    element.MagnitudePattern = g;
    element.PhasePattern = zeros(size(g));
end

function element = reflectorDipoleElement(fq)
    %reflectorDipoleElement   Design reflector-backed dipole antenna element
    
    % Design reflector and exciter, which is vertical dipole by default
    element = design(reflector,fq);
    element.Exciter = design(element.Exciter,fq);
    
    % Tilt antenna element to radiate in xy-plane, with boresight along x-axis
    element.Tilt = 90;
    element.TiltAxis = "y";
    element.Exciter.Tilt = 90;
    element.Exciter.TiltAxis = "y";
end