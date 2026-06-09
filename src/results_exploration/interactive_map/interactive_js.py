"""
JavaScript interactivity module for adding interactive functionality to the map.
"""

import folium

def add_edge_toggle_javascript(m):
    """
    Add JavaScript code for edge toggling functionality.
    
    Args:
        m (folium.Map): The map to add JavaScript to
    """
    js_code = '''
    <script>
    // Global state for visibility (keys are node indices)
    const outgoingVisible = {};
    const incomingVisible = {};

    function updateEdgesVisibility() {
        const edges = document.querySelectorAll('.edge');
        edges.forEach(edge => {
            const classString = typeof edge.className === 'string'
                ? edge.className
                : (edge.className && edge.className.baseVal ? edge.className.baseVal : "");
            const classes = classString.split(" ");
            let src = null, tgt = null;
            classes.forEach(cls => {
                if (cls.startsWith("source_")) {
                    src = cls.replace("source_", "");
                }
                if (cls.startsWith("target_")) {
                    tgt = cls.replace("target_", "");
                }
            });
            
            // Initialize visibility state if not set yet
            if (src !== null && !(src in outgoingVisible)) {
                outgoingVisible[src] = true;
            }
            if (tgt !== null && !(tgt in incomingVisible)) {
                incomingVisible[tgt] = true;
            }
            
            if (src !== null && tgt !== null && outgoingVisible[src] && incomingVisible[tgt]) {
                edge.style.visibility = "visible";
            } else {
                edge.style.visibility = "hidden";
            }
        });
    }

    function toggleOutgoingPaths(nodeId) {
        outgoingVisible[nodeId] = !outgoingVisible[nodeId];
        console.log("Toggled outgoing for node", nodeId, outgoingVisible[nodeId]);
        updateEdgesVisibility();
    }

    function toggleIncomingPaths(nodeId) {
        incomingVisible[nodeId] = !incomingVisible[nodeId];
        console.log("Toggled incoming for node", nodeId, incomingVisible[nodeId]);
        updateEdgesVisibility();
    }

    function toggleNodePaths(nodeId) {
        outgoingVisible[nodeId] = !outgoingVisible[nodeId];
        incomingVisible[nodeId] = !incomingVisible[nodeId];
        console.log("Toggled all for node", nodeId, "Outgoing:", outgoingVisible[nodeId], "Incoming:", incomingVisible[nodeId]);
        updateEdgesVisibility();
    }
    
    // Initialize visibility map after the page is loaded
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(updateEdgesVisibility, 1000);
    });
    </script>
    '''
    m.get_root().html.add_child(folium.Element(js_code))

def add_node_highlight_javascript(m):
    """
    Add JavaScript code for highlighting connected nodes when hovering over a node.
    
    Args:
        m (folium.Map): The map to add JavaScript to
    """
    js_code = '''
    <script>
    // Store all nodes and edges
    const allNodes = {};
    const connectedEdges = {};
    
    // Function to collect all nodes and their connections
    function initializeNodeConnections() {
        // Collect all nodes
        document.querySelectorAll('.node-marker').forEach(node => {
            const nodeId = node.getAttribute('data-node-id');
            if (nodeId) {
                allNodes[nodeId] = node;
            }
        });
        
        // Collect all edges and their connections
        document.querySelectorAll('.edge').forEach(edge => {
            const classString = typeof edge.className === 'string'
                ? edge.className
                : (edge.className && edge.className.baseVal ? edge.className.baseVal : "");
            const classes = classString.split(" ");
            
            let src = null, tgt = null;
            classes.forEach(cls => {
                if (cls.startsWith("source_")) {
                    src = cls.replace("source_", "");
                }
                if (cls.startsWith("target_")) {
                    tgt = cls.replace("target_", "");
                }
            });
            
            if (src && tgt) {
                // Add this edge to the source node's outgoing connections
                if (!connectedEdges[src]) {
                    connectedEdges[src] = {outgoing: [], incoming: []};
                }
                connectedEdges[src].outgoing.push({edge: edge, target: tgt});
                
                // Add this edge to the target node's incoming connections
                if (!connectedEdges[tgt]) {
                    connectedEdges[tgt] = {outgoing: [], incoming: []};
                }
                connectedEdges[tgt].incoming.push({edge: edge, source: src});
            }
        });
        
        console.log("Initialized node connections", Object.keys(allNodes).length, "nodes");
    }
    
    // Highlight connected nodes and edges
    function highlightConnections(nodeId) {
        if (!connectedEdges[nodeId]) return;
        
        // Highlight outgoing connections
        connectedEdges[nodeId].outgoing.forEach(conn => {
            conn.edge.style.strokeWidth = '4px';
            conn.edge.style.stroke = '#ffff00'; // Yellow
            if (allNodes[conn.target]) {
                allNodes[conn.target].style.stroke = '#ffff00';
                allNodes[conn.target].style.strokeWidth = '2px';
            }
        });
        
        // Highlight incoming connections
        connectedEdges[nodeId].incoming.forEach(conn => {
            conn.edge.style.strokeWidth = '4px';
            conn.edge.style.stroke = '#00ffff'; // Cyan
            if (allNodes[conn.source]) {
                allNodes[conn.source].style.stroke = '#00ffff';
                allNodes[conn.source].style.strokeWidth = '2px';
            }
        });
    }
    
    // Reset highlights
    function resetHighlights() {
        // Reset all edges
        document.querySelectorAll('.edge').forEach(edge => {
            edge.style.strokeWidth = '';
            edge.style.stroke = '';
        });
        
        // Reset all nodes
        for (const nodeId in allNodes) {
            allNodes[nodeId].style.stroke = '';
            allNodes[nodeId].style.strokeWidth = '';
        }
    }
    
    // Initialize when the DOM is loaded
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(initializeNodeConnections, 1500);
        
        // Add hover event listeners to nodes
        document.querySelectorAll('.node-marker').forEach(node => {
            node.addEventListener('mouseenter', function() {
                const nodeId = this.getAttribute('data-node-id');
                if (nodeId) {
                    highlightConnections(nodeId);
                }
            });
            
            node.addEventListener('mouseleave', function() {
                resetHighlights();
            });
        });
    });
    </script>
    '''
    m.get_root().html.add_child(folium.Element(js_code))

def add_global_edge_toggle_javascript(m):
    """
    Add JavaScript code for toggling all edges on the map with a global button.
    
    Args:
        m (folium.Map): The map to add JavaScript to
    """
    js_code = '''
    <script>
    // Global state for all edges visibility
    let allEdgesVisible = true;
    
    // Function to toggle all edges on the map
    function toggleAllEdges() {
        allEdgesVisible = !allEdgesVisible;
        
        // Apply the visibility to all edges
        const edges = document.querySelectorAll('.edge');
        edges.forEach(edge => {
            edge.style.visibility = allEdgesVisible ? "visible" : "hidden";
        });
        
        // Update button text
        const toggleButton = document.getElementById('toggle-all-edges-btn');
        if (toggleButton) {
            if (allEdgesVisible) {
                toggleButton.textContent = "Hide Edges";
            } else {
                toggleButton.textContent = "Show Edges";
            }
        }
        
        console.log("Toggled all edges:", allEdgesVisible);
    }
    
    // Add a fixed control to the map for the toggle button
    const toggleAllEdgesControl = L.Control.extend({
        options: {
            position: 'topright'
        },
        
        onAdd: function() {
            const container = L.DomUtil.create('div', 'leaflet-bar leaflet-control');
            container.style.padding = '0';
            container.style.margin = '30px 0px 0 0';
            
            const button = L.DomUtil.create('a', '', container);
            button.id = 'toggle-all-edges-btn';
            button.href = '#';
            button.style.display = 'flex';
            button.style.alignItems = 'center';
            button.style.justifyContent = 'center';
            button.style.width = '110px';
            button.style.padding = '8px 12px';
            button.style.fontFamily = '"Segoe UI", Roboto, Arial, sans-serif';
            button.style.fontSize = '13px';
            button.style.fontWeight = '600';
            button.style.textDecoration = 'none';
            button.style.cursor = 'pointer';
            button.style.borderRadius = '4px';
            button.style.transition = 'all 0.3s ease';
            button.textContent = "Hide Edges";
            
            // Add hover effect but maintain default styling
            button.onmouseover = function() {
                this.style.backgroundColor = '#f4f4f4';
            };
            button.onmouseout = function() {
                this.style.backgroundColor = '';
            };
            
            L.DomEvent
                .on(button, 'click', L.DomEvent.preventDefault)
                .on(button, 'click', toggleAllEdges);
            
            return container;
        }
    });
    
    // Add the control to the map and position it relative to the layers control
    window.addEventListener('load', function() {
        setTimeout(() => {
            try {
                // Find the map object
                let mapObj;
                if (typeof map !== 'undefined') {
                    mapObj = map;
                } else {
                    // Try to get the map from folium's internals
                    const maps = Object.values(window).filter(o => o instanceof L.Map);
                    if (maps.length > 0) {
                        mapObj = maps[0];
                    } else {
                        console.error("Could not find map object to add control");
                        return;
                    }
                }
                
                // Add the toggle button control
                const edgeToggleControl = new toggleAllEdgesControl().addTo(mapObj);
                
                // Position it below the layers control (which is automatically added by folium)
                setTimeout(() => {
                    try {
                        // Find the layers control container
                        const layersControl = document.querySelector('.leaflet-control-layers');
                        if (layersControl) {
                            // Get the toggle button container
                            const toggleButton = document.getElementById('toggle-all-edges-btn');
                            if (toggleButton && toggleButton.parentElement) {
                                // Calculate position to place it below layers control
                                toggleButton.parentElement.style.position = 'absolute';
                                toggleButton.parentElement.style.top = (layersControl.offsetHeight + 40) + 'px';
                                toggleButton.parentElement.style.right = '10px';
                                toggleButton.parentElement.style.zIndex = '1000';
                            }
                        }
                    } catch (e) {
                        console.error("Error positioning toggle button:", e);
                    }
                }, 500);
                
                console.log("Edge toggle button added to map");
            } catch (e) {
                console.error("Error adding edge toggle control:", e);
            }
        }, 1000);
    });
    </script>
    '''
    m.get_root().html.add_child(folium.Element(js_code))


def add_all_interactive_features(m):
    """
    Add all interactive features to the map.
    
    Args:
        m (folium.Map): The map to add interactive features to
    """
    add_edge_toggle_javascript(m)
    add_node_highlight_javascript(m)
    add_global_edge_toggle_javascript(m) 