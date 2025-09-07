// D3.js visualization script
const svgElement = document.getElementById("workflow-graph");
const width = svgElement.clientWidth;
const height = svgElement.clientHeight;
const svg = d3.select(svgElement);

// Simple Python syntax highlighter
function highlightPythonSyntax(code) {
    if (!code) return code;

    // First escape any HTML to prevent XSS and unwanted HTML rendering
    const escapeHTML = str => str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');

    // Escape the code first
    const escapedCode = escapeHTML(code);

    // Replace Python keywords and other syntax elements with highlighted spans
    return escapedCode
        // Keywords
        .replace(/\b(def|class|import|from|return|if|else|elif|for|while|try|except|finally|with|as|in|is|not|and|or|True|False|None)\b/g,
                '<span class="keyword">$1</span>')
        // Function calls
        .replace(/(\w+)(\s*\()/g, '<span class="function">$1</span>$2')
        // Strings
        .replace(/(&quot;(?:[^&]|&amp;)*?&quot;)|(&apos;(?:[^&]|&amp;)*?&apos;)/g, '<span class="string">$1$2</span>')
        // Numbers
        .replace(/\b(\d+(\.\d+)?)\b/g, '<span class="number">$1</span>')
        // Comments
        .replace(/(#.*)$/gm, '<span class="comment">$1</span>');
}

// Create a container group for all elements that will be transformed
const container = svg.append("g")
    .attr("class", "zoom-container");

// Add zoom behavior
const zoom = d3.zoom()
    .scaleExtent([0.1, 4]) // Min/max zoom level
    .on("zoom", (event) => {
        container.attr("transform", event.transform);
    });

// Apply zoom behavior to svg
svg.call(zoom)
    .call(zoom.transform, d3.zoomIdentity); // Start with identity transform

// Add zoom controls
const zoomControls = svg.append("g")
    .attr("class", "zoom-controls")
    .attr("transform", "translate(20, 20)"); // Position in top-left corner

// Zoom in button
zoomControls.append("rect")
    .attr("class", "zoom-btn")
    .attr("x", 0)
    .attr("y", 0)
    .attr("width", 30)
    .attr("height", 30)
    .attr("rx", 5)
    .attr("fill", "white")
    .attr("stroke", "#6b7280")
    .style("cursor", "pointer")
    .on("click", () => {
        svg.transition()
            .duration(300)
            .call(zoom.scaleBy, 1.3);
    });

zoomControls.append("text")
    .attr("x", 15)
    .attr("y", 20)
    .attr("text-anchor", "middle")
    .attr("fill", "#4b5563")
    .style("font-size", "20px")
    .style("pointer-events", "none")
    .text("+");

// Zoom out button
zoomControls.append("rect")
    .attr("class", "zoom-btn")
    .attr("x", 0)
    .attr("y", 40)
    .attr("width", 30)
    .attr("height", 30)
    .attr("rx", 5)
    .attr("fill", "white")
    .attr("stroke", "#6b7280")
    .style("cursor", "pointer")
    .on("click", () => {
        svg.transition()
            .duration(300)
            .call(zoom.scaleBy, 0.7);
    });

zoomControls.append("text")
    .attr("x", 15)
    .attr("y", 60)
    .attr("text-anchor", "middle")
    .attr("fill", "#4b5563")
    .style("font-size", "20px")
    .style("pointer-events", "none")
    .text("−");

// Reset zoom button
zoomControls.append("rect")
    .attr("class", "zoom-btn")
    .attr("x", 0)
    .attr("y", 80)
    .attr("width", 30)
    .attr("height", 30)
    .attr("rx", 5)
    .attr("fill", "white")
    .attr("stroke", "#6b7280")
    .style("cursor", "pointer")
    .on("click", () => {
        svg.transition()
            .duration(300)
            .call(zoom.transform, d3.zoomIdentity);
    });

zoomControls.append("text")
    .attr("x", 15)
    .attr("y", 100)
    .attr("text-anchor", "middle")
    .attr("fill", "#4b5563")
    .style("font-size", "14px")
    .style("pointer-events", "none")
    .text("⟲");

// Define arrow markers for directed links
svg.append("defs").append("marker")
    .attr("id", "arrowhead")
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 20)
    .attr("refY", 0)
    .attr("markerWidth", 6)
    .attr("markerHeight", 6)
    .attr("orient", "auto")
    .append("path")
    .attr("d", "M0,-5L10,0L0,5")
    .attr("fill", "#6b7280");

// Modal functionality
function createModal() {
    const modalHtml = `
        <div id="metadataModal" class="hidden fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
            <div class="relative top-20 mx-auto p-5 border w-11/12 md:w-3/4 lg:w-1/2 shadow-lg rounded-md bg-white">
                <div class="flex justify-between items-center">
                    <h2 class="text-xl font-bold text-gray-700">Node Metadata</h2>
                    <button class="close-modal text-gray-400 bg-transparent hover:bg-gray-200 hover:text-gray-900 rounded-lg text-sm w-8 h-8 flex justify-center items-center">
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                        </svg>
                    </button>
                </div>
                <div class="modal-body mt-4 text-gray-600"></div>
            </div>
        </div>
    `;
    document.body.insertAdjacentHTML('beforeend', modalHtml);
}

function showModal(content) {
    const modal = document.getElementById('metadataModal');
    const modalBody = modal.querySelector('.modal-body');
    modalBody.innerHTML = content;
    modal.classList.remove('hidden');
}

function setupModalEvents() {
    const modal = document.getElementById('metadataModal');
    const closeBtn = modal.querySelector('.close-modal');

    closeBtn.onclick = () => modal.classList.add('hidden');
    window.onclick = (event) => {
        if (event.target === modal) {
            modal.classList.add('hidden');
        }
    };
}

// Create modal on page load
createModal();
setupModalEvents();

// Tooltip functionality
const tooltip = d3.select("#tooltip");

function getNodeTooltipContent(d, isTooltip = false) {
    let content = `<div class="space-y-2">
        <div><span class="font-semibold">ID:</span> ${d.id}</div>
        <div><span class="font-semibold">Type:</span> ${d.label}</div>`;

    // Show map-specific information
    if (d.metadata && (d.label === "map" || (d.metadata.belongs_to_node && !isTooltip))) {
        content += `<div class="mt-2 border-t pt-2">
            <div class="font-semibold text-blue-600">Composite Operation</div>`;

        if (d.label === "map") {
            content += `<div><span class="font-semibold">Iterates On:</span> ${d.metadata.iterate_on || "N/A"}</div>
                <div><span class="font-semibold">Iterate As:</span> ${d.metadata.iterate_as || "N/A"}</div>`;
        }

        if (d.metadata.belongs_to_node) {
            content += `<div><span class="font-semibold">Part of Node:</span> ${d.metadata.belongs_to_node}</div>`;
        }

        content += `</div>`;
    }

    // Show parallel-specific information
    if (d.metadata && (d.label === "parallel" || (d.metadata.node_type === "parallel"))) {
        content += `<div class="mt-2 border-t pt-2">
            <div class="font-semibold text-purple-600">Parallel Operation</div>`;

        if (d.metadata.node_type === "parallel") {
            content += `<div><span class="font-semibold">Parallel Group ID:</span> ${d.metadata.parallel_branch_id || "N/A"}</div>`;
        }

        content += `</div>`;
    }

    // Show task subtype for task nodes
    if (d.task_type) {
        content += `<div><span class="font-semibold">Task Subtype:</span> ${d.task_type}</div>`;
    }

    // For tooltip, only show basic info (ID, Type, Subtype)
    if (isTooltip) {
        content += `</div>`;
        return content;
    }

    // Show condition for conditional nodes
    if (d.label === 'conditional' && d.condition) {
        content += `
        <div class="mt-4">
            <div class="font-semibold mb-2">Condition:</div>
            <div class="pl-4 bg-gray-50 p-2 rounded">
                <code class="text-sm">${d.condition}</code>
            </div>
        </div>`;
    }

    // Show display string if present in metadata
    if (d.metadata && d.metadata.display) {
        content += `
        <div class="mt-4">
            <div class="font-semibold mb-2">Display:</div>
            <pre class="whitespace-pre-wrap bg-gray-100 p-2 rounded text-sm">${d.metadata.display}</pre>
        </div>`;
    }

    // Show function signature for Python tasks (only in modal, not in tooltip)
    if (!isTooltip && d.label === 'task' && d.task_type === 'python' && d.metadata && d.metadata.signature) {
        // Apply syntax highlighting to the code
        const highlightedCode = highlightPythonSyntax(d.metadata.signature);
        content += `
        <div class="mt-4">
            <div class="font-semibold mb-2">Function:</div>
            <div class="code-block">
                <div class="code-header">python</div>
                <pre class="language-python whitespace-pre-wrap bg-gray-800 p-3 rounded-b text-sm overflow-x-auto font-mono text-gray-100">${highlightedCode}</pre>
            </div>
        </div>`;
    }

    // Show metadata if available
    if (d.metadata && Object.keys(d.metadata).length > 0) {
        content += `
        <div class="mt-4">
            <div class="font-semibold mb-2">Metadata:</div>
            <div class="pl-4 space-y-1">`;
        for (const key in d.metadata) {
            let value = d.metadata[key];
            // Don't duplicate display string or signature
            if (key === 'display' || key === 'signature') continue;

            // Pretty print objects and arrays
            if (typeof value === 'object') {
                const jsonStr = JSON.stringify(value, null, 2);
                content += `
                <div class="mb-2">
                    <span class="text-gray-700 font-semibold">${key}:</span>
                    <div class="code-block mt-1">
                        <div class="code-header">json</div>
                        <pre class="whitespace-pre-wrap bg-gray-800 p-2 rounded-b text-sm overflow-x-auto font-mono text-gray-100">${jsonStr}</pre>
                    </div>
                </div>`;
            } else {
                content += `<div class="mb-1"><span class="text-gray-700 font-semibold">${key}:</span> <code class="bg-gray-100 px-1 py-0.5 rounded text-sm">${value}</code></div>`;
            }
        }
        content += `</div></div>`;
    }

    content += `</div>`;
    return content;
}

function renderGraph(graphData) {
    // Clear existing graph
    container.selectAll("*").remove();

    const nodeIds = new Set(graphData.nodes.map(d => d.id));
    const validLinks = graphData.links.filter(
        link => nodeIds.has(link.source) && nodeIds.has(link.target)
    );

    // A simple topological sort to order nodes for a left-to-right layout
    const nodeMap = new Map(graphData.nodes.map(d => [d.id, { ...d, dependencies: 0, dependents: 0 }]));
    validLinks.forEach(link => {
        nodeMap.get(link.target).dependencies++;
        nodeMap.get(link.source).dependents++;
    });

    // Identify start nodes (no incoming links)
    let startNodes = Array.from(nodeMap.values()).filter(d => d.dependencies === 0);
    let queue = [...startNodes];
    const sortedNodes = [];
    const visited = new Set();
    const levels = new Map();
    let currentLevel = 0;

    while (queue.length > 0) {
        const levelSize = queue.length;
        for (let i = 0; i < levelSize; i++) {
            const node = queue.shift();
            if (!visited.has(node.id)) {
                visited.add(node.id);
                sortedNodes.push(node);
                levels.set(node.id, currentLevel);

                validLinks.filter(link => link.source === node.id).forEach(link => {
                    queue.push(nodeMap.get(link.target));
                });
            }
        }
        currentLevel++;
    }

    // Map the sorted nodes back to the original array to maintain order
    const sortedGraphDataNodes = graphData.nodes.sort((a, b) => levels.get(a.id) - levels.get(b.id));

    // Identify parallel branches for better positioning
    const parallelBranches = new Map(); // Maps node id to its branch path
    const branchY = new Map(); // Maps branch path to a y-position

    // Find all branch paths in the graph
    sortedGraphDataNodes.forEach(node => {
        if (node.id) {
            // Extract branch path from node id (format often includes branch names)
            const parts = node.id.split('.');
            if (parts.length > 1) {
                const branchPath = parts.slice(0, -1).join('.');
                parallelBranches.set(node.id, branchPath);

                // If this is a new branch, assign it a y-position
                if (!branchY.has(branchPath)) {
                    branchY.set(branchPath, branchY.size);
                }
            }
        }
    });

    // Set up force simulation with improved positioning for parallel branches
    const simulation = d3.forceSimulation(sortedGraphDataNodes)
        .force("link", d3.forceLink(validLinks).id(d => d.id).distance(100))
        .force("charge", d3.forceManyBody().strength(-300))
        .force("x", d3.forceX(d => {
            const level = levels.get(d.id);
            const totalLevels = Math.max(...Array.from(levels.values())) + 1;
            return (width / totalLevels) * (level + 0.5);
        }).strength(0.5))
        // Add additional forces for nodes within the same map group to stay close together
        .force("map-group", d => {
            if (graphData.mapGroups && d.metadata && d.metadata.belongs_to_node) {
                // Find related nodes in the same map group and pull them closer
                const mapGroup = graphData.mapGroups.find(g => g.id === d.metadata.belongs_to_node);
                if (mapGroup) {
                    // Add an additional force to keep map-related nodes together
                    return d3.forceY().strength(0.2);
                }
            }
            return null;
        })
        // Add special force to push parallel next nodes away from the parallel group
        .force("parallel-next", d => {
            // If this node is a next node of a parallel node, push it away
            if (graphData.mapGroups) {
                // Check if this node is a next node of any parallel node
                const parallelGroups = graphData.mapGroups.filter(g =>
                    g.type === 'parallel' &&
                    g.nextNodeIds &&
                    g.nextNodeIds.includes(d.id)
                );

                if (parallelGroups.length > 0) {
                    // This is a next node of a parallel group
                    // Create a much stronger force to push it far to the right

                    // Find the source parallel node level
                    const parentId = parallelGroups[0].id;
                    const parentLevel = levels.get(parentId) || 0;

                    // Force this node to be at least two levels to the right of its parent
                    const targetLevel = parentLevel + 2;
                    const targetX = (width / (Math.max(...Array.from(levels.values())) + 3)) * (targetLevel + 0.5);

                    // Add a Y force to position it near the middle vertically
                    d3.forceY(height / 2).strength(0.5)(d);

                    // Use a maximum strength force to ensure separation
                    return d3.forceX(targetX).strength(2.0);
                }
            }
            return null;
        })
        .force("y", d3.forceY(d => {
            // Position nodes based on their branch
            const branchPath = parallelBranches.get(d.id);
            if (branchPath && branchY.has(branchPath)) {
                // Calculate y position based on branch index
                const branchCount = branchY.size;
                const branchIndex = branchY.get(branchPath);
                // Distribute branches evenly in the vertical space
                return (height / (branchCount + 1)) * (branchIndex + 1);
            }
            return height / 2; // Default center position
        }).strength(0.3));

    // Define marker for failure links - add red X marker
    svg.append("defs")
        .append("marker")
        .attr("id", "failure-marker")
        .attr("viewBox", "0 0 10 10")
        .attr("refX", 5)
        .attr("refY", 5)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M 1,1 L 9,9 M 9,1 L 1,9")
        .attr("stroke", "#e53e3e")
        .attr("stroke-width", 1.5);

    // Define a special marker for parallel next links
    svg.append("defs")
        .append("marker")
        .attr("id", "parallel-arrow")
        .attr("viewBox", "0 0 10 10")
        .attr("refX", 5)
        .attr("refY", 5)
        .attr("markerWidth", 8)  // Slightly larger
        .attr("markerHeight", 8) // Slightly larger
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M 0,0 L 10,5 L 0,10 Z") // Triangle
        .attr("fill", "#9333ea"); // Purple to match parallel node color

    // Create links with differentiation between success and failure paths
    // Use paths instead of lines for more flexibility with parallel node connections
    const link = container.append("g")
        .attr("class", "links")
        .selectAll("path")
        .data(graphData.links)
        .enter()
        .append("path")
        .attr("stroke", d => {
            // Check if this is a connection from a parallel node to its next node
            const isParallelNextLink = graphData.mapGroups &&
                graphData.mapGroups.some(group =>
                    group.type === 'parallel' &&
                    group.id === d.source &&
                    group.nextNodeIds &&
                    group.nextNodeIds.includes(d.target));

            if (isParallelNextLink) return "#9333ea"; // Bright purple for parallel next connection
            if (d.type === "failure") return "#e53e3e"; // Red for failure
            if (d.type === "success") return "#38a169"; // Green for success
            return "#9ca3af"; // Gray for default/other
        })
        .attr("fill", "none") // Important for paths
        .attr("stroke-width", d => {
            // Check if this is a connection from a parallel node to its next node
            const isParallelNextLink = graphData.mapGroups &&
                graphData.mapGroups.some(group =>
                    group.type === 'parallel' &&
                    group.id === d.source &&
                    group.nextNodeIds &&
                    group.nextNodeIds.includes(d.target));

            return isParallelNextLink ? 2.5 : (d.type === "failure" || d.type === "success" ? 2 : 1.5);
        })
        .attr("stroke-dasharray", d => {
            const targetNode = sortedGraphDataNodes.find(n => n.id === d.target);
            const isBranchLink = targetNode?.metadata?.belongs_to_node === d.source;

            if (isBranchLink) return "4,4"; // Dashed for branch links

            if (d.type === "failure") return "5,5"; // Dashed for failure paths
            return "none";
        })
        .attr("marker-end", d => {
            if (d.type === "failure") return "url(#failure-marker)";
            return "url(#arrowhead)";
        })
        .attr("stroke-opacity", 0.8);

    // Create Nodes as Images with text labels
    const node = container.append("g")
        .attr("class", "nodes")
        .selectAll("g")
        .data(sortedGraphDataNodes)
        .enter().append("g")
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended))
        .on("click", (event, d) => {
            // Prevent modal from opening if we're dragging
            if (event.defaultPrevented) return;
            // Show modal with detailed metadata including signature
            const content = getNodeTooltipContent(d, false); // false = not tooltip, show signature
            showModal(content);
        });

    // Append IMAGES for nodes
    node.append("image")
        .attr("xlink:href", d => {
            // Use relative paths to your local images
            if (d.label === "conditional") return "static/images/conditional.png";
            if (d.label === "map") return "static/images/parallel.png"; // Use parallel icon for map for now
            if (d.task_type === "python") return "static/images/python.png";
            if (d.task_type === "notebook") return "static/images/notebook.png";
            if (d.task_type === "shell") return "static/images/shell.png";
            if (d.label === "parallel") return "static/images/parallel.png";
            if (d.label === "success") return "static/images/success.png";
            if (d.label === "fail") return "static/images/failure.png";
            if (d.label === "stub") return "static/images/stub.png";
            // Provide a default image for unknown types
            return "static/images/python.png";
        })
        .attr("width", 32)
        .attr("height", 32)
        .attr("class", d => {
            // Log node info for debugging
            console.log("Processing node:", d);

            let classes = `node-image ${d.label} ${d.task_type ? 'task-' + d.task_type : ''}`;

            // Add special styling for map nodes and their branch nodes
            if (d.label === "map" || (d.metadata && d.metadata.node_type === "map")) {
                classes += ' map-root-node';
                console.log("Adding map-root-node class to:", d.id);
            }

            // Add special styling for parallel nodes
            if (d.label === "parallel" || (d.metadata && d.metadata.node_type === "parallel")) {
                classes += ' parallel-root-node';
                console.log("Adding parallel-root-node class to:", d.id);
            }

            if (d.metadata && d.metadata.belongs_to_node) {
                classes += ' map-branch-node';
                console.log("Adding map-branch-node class to:", d.id);

                // If belongs_to_node has a parallel_ prefix, add special class
                if (d.metadata.belongs_to_node.startsWith('parallel_')) {
                    classes += ' parallel-branch-node';
                    console.log("Adding parallel-branch-node class to:", d.id);
                }
            }

            return classes;
        });

    // Add a decorative border for map nodes
    node.filter(d => d.label === "map" || (d.metadata && d.metadata.node_type === "map"))
        .append("circle")
        .attr("r", 20)
        .attr("fill", "none")
        .attr("stroke", "#3b82f6")  // Blue for map nodes
        .attr("stroke-width", 2)
        .attr("opacity", 0.8)
        .attr("stroke-dasharray", "4,2");

    // Add a decorative border for parallel nodes
    node.filter(d => d.label === "parallel" || (d.metadata && d.metadata.node_type === "parallel"))
        .append("circle")
        .attr("r", 20)
        .attr("fill", "none")
        .attr("stroke", "#9333ea")  // Purple for parallel nodes
        .attr("stroke-width", 2)
        .attr("opacity", 0.8)
        .attr("stroke-dasharray", "4,2");

    // Add a small indicator for nodes that are part of a map branch (blue)
    node.filter(d => d.metadata && d.metadata.belongs_to_node &&
                    !d.metadata.belongs_to_node.startsWith('parallel_') &&
                    d.label !== "map")
        .append("circle")
        .attr("r", 5)
        .attr("cx", 16)
        .attr("cy", -10)
        .attr("fill", "#3b82f6") // Blue for map branches
        .attr("opacity", 0.7);

    // Add a small indicator for nodes that are part of a parallel branch (purple)
    node.filter(d => d.metadata && d.metadata.belongs_to_node &&
                    d.metadata.belongs_to_node.startsWith('parallel_') &&
                    d.label !== "parallel")
        .append("circle")
        .attr("r", 5)
        .attr("cx", 16)
        .attr("cy", -10)
        .attr("fill", "#9333ea") // Purple for parallel branches
        .attr("opacity", 0.7);

    // Add a text label below the image
    node.append("text")
        .attr("dy", "2.5em")
        .attr("text-anchor", "middle")
        .text(d => d.alias || d.id) // Use alias as display name if available, fallback to id
        .attr("font-size", "10px")
        .attr("fill", "#4b5563")
        .attr("pointer-events", "none");

    // Create decorative boxes around map branches if map groups exist
    if (!graphData.mapGroups) {
        console.log("No map groups found, initializing empty array");
        graphData.mapGroups = [];
    }

    if (graphData.mapGroups.length > 0) {
        console.log("Creating map group boxes for:", graphData.mapGroups);

        // Add group containers for map branches
        const mapGroups = container.append("g")
            .attr("class", "map-groups")
            .selectAll("g")
            .data(graphData.mapGroups)
            .enter()
            .append("g")
            .attr("class", "map-group")
            .attr("id", d => `map-group-${d.id.replace(/\./g, '-')}`);

        // Add rectangle around each map/parallel/conditional group
        const mapBoxes = mapGroups.append("rect")
            .attr("class", d => {
                if (d.type === 'parallel') return "parallel-box";
                if (d.type === 'conditional') return "conditional-box";
                return "map-box";
            })
            .attr("rx", 8) // Rounded corners
            .attr("ry", 8)
            .attr("fill", "none")
            .attr("stroke", d => {
                if (d.type === 'parallel') return "#9333ea"; // Purple for parallel
                if (d.type === 'conditional') return "#f59e0b"; // Orange for conditional
                return "#3b82f6"; // Blue for map
            })
            .attr("stroke-width", 1.5)
            .attr("stroke-dasharray", "5,3") // Dashed line
            .attr("opacity", 0.7)
            .attr("pointer-events", "none"); // Don't interfere with clicks

        // Add a label for the map/parallel/conditional operation
        mapGroups.append("text")
            .attr("class", d => {
                if (d.type === 'parallel') return "parallel-label";
                if (d.type === 'conditional') return "conditional-label";
                return "map-label";
            })
            .attr("fill", d => {
                if (d.type === 'parallel') return "#9333ea"; // Purple for parallel
                if (d.type === 'conditional') return "#f59e0b"; // Orange for conditional
                return "#3b82f6"; // Blue for map
            })
            .attr("font-size", "12px")
            .attr("font-weight", "bold")
            .text(d => {
                console.log("Creating label for group:", d);
                if (d.type === 'parallel') {
                    return `Branch: ${d.branchName}`;
                } else if (d.type === 'conditional') {
                    return `Conditional Branch: ${d.branchName}`;
                } else {
                    // For map nodes
                    return `Map Branch: ${d.branchName}`;
                }
            })
            .attr("pointer-events", "none");

        // Special connections are handled through the existing links to preserve the simulation's physics

        // Update the positions of the map boxes during simulation
        simulation.on("tick.mapboxes", () => {
            mapGroups.each(function(d) {
                const group = d3.select(this);
                // Find positions of all nodes in this map group
                const nodes = d.nodes.map(node => ({
                    x: node.x,
                    y: node.y
                }));

                if (nodes.length > 0) {
                    // Calculate bounding box
                    const minX = d3.min(nodes, node => node.x) - 25;
                    const minY = d3.min(nodes, node => node.y) - 25;
                    const maxX = d3.max(nodes, node => node.x) + 25;
                    const maxY = d3.max(nodes, node => node.y) + 35;
                    const width = Math.max(maxX - minX, 100); // Minimum width
                    const height = Math.max(maxY - minY, 100); // Minimum height

                    // Update rectangle position and size
                    group.select("rect")
                        .attr("x", minX)
                        .attr("y", minY)
                        .attr("width", width)
                        .attr("height", height);

                    // Update label position
                    group.select("text")
                        .attr("x", minX + 10)
                        .attr("y", minY - 10);
                }
            });
        });
    }

    // Tooltip functionality
    node.on("mouseover", function(event, d) {
        tooltip.html(getNodeTooltipContent(d, true)) // true = is tooltip, don't show signature
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 20) + "px")
            .style("display", "block")
            .style("opacity", 1);
    })
    .on("mouseout", function() {
        tooltip.style("opacity", 0)
            .transition()
            .delay(200)
            .style("display", "none");
    });

    // Add tooltips for links to show their type
    link.append("title")
        .text(d => {
            if (d.type === "failure") return "Failure Path";
            if (d.type === "success") return "Success Path";
            return "Default Path";
        });

    // Update positions on each simulation tick
    simulation.on("tick", () => {
        link.attr("d", d => {
            const sourceX = d.source.x;
            const sourceY = d.source.y;
            const targetX = d.target.x;
            const targetY = d.target.y;

            const dx = targetX - sourceX;
            const dy = targetY - sourceY;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist === 0) return `M ${sourceX} ${sourceY}`; // Avoid division by zero
            const r = 16; // Node radius

            const sourceOffsetX = sourceX + (dx * r) / dist;
            const sourceOffsetY = sourceY + (dy * r) / dist;
            const targetOffsetX = targetX - (dx * r) / dist;
            const targetOffsetY = targetY - (dy * r) / dist;

            return `M ${sourceOffsetX} ${sourceOffsetY} L ${targetOffsetX} ${targetOffsetY}`;
        });

        node.attr("transform", d => `translate(${d.x},${d.y})`);
    });

    // Drag functions
    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
    }

    // Update text colors to match current theme after rendering
    updateSVGTextColors();
}

// Fetch graph data from the API and render the graph
async function fetchAndRenderGraph(tempPath, functionName) {
    try {
        const response = await fetch(`/graph-data?temp_path=${encodeURIComponent(tempPath)}&function=${encodeURIComponent(functionName)}`);
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail);
        }
        const graphData = await response.json();

        // Process map nodes for visualization
        processCompositeNodeData(graphData);

        renderGraph(graphData);
    } catch (error) {
        console.error('There has been a problem with your fetch operation:', error);

        // Display error in UI
        let errorElement = document.getElementById('graph-error');
        if (!errorElement) {
            errorElement = document.createElement('div');
            errorElement.id = 'graph-error';
            errorElement.className = 'text-red-600 mt-4 p-4 bg-red-100 rounded';
            document.getElementById('loadGraphBtn').insertAdjacentElement('afterend', errorElement);
        }
        errorElement.textContent = error.message;
        errorElement.style.display = 'block';
        throw error; // Re-throw the error so the finally block in the click handler can hide the spinner
    }
}

// Create loading spinner element
const loadingSpinner = document.createElement('div');
loadingSpinner.id = 'loading-spinner';
loadingSpinner.className = 'hidden fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2';
loadingSpinner.innerHTML = `
    <div class="flex items-center justify-center">
        <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
        <span class="ml-3 text-gray-700">Loading graph...</span>
    </div>
`;
document.body.appendChild(loadingSpinner);

let currentTempPath = null;

// File upload handler
document.getElementById('uploadBtn').addEventListener('click', async () => {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    const uploadStatus = document.getElementById('uploadStatus');

    if (!file) {
        uploadStatus.textContent = 'Please select a file to upload.';
        uploadStatus.className = 'text-sm text-red-600';
        return;
    }

    if (!file.name.endsWith('.py')) {
        uploadStatus.textContent = 'Please upload a Python (.py) file.';
        uploadStatus.className = 'text-sm text-red-600';
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        uploadStatus.textContent = 'Uploading...';
        uploadStatus.className = 'text-sm text-blue-600';

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
            currentTempPath = result.temp_path;
            uploadStatus.textContent = `File uploaded successfully: ${result.filename}`;
            uploadStatus.className = 'text-sm text-green-600';

            // Update the header with the file name
            const fileNameDisplay = document.getElementById('file-name-display');
            fileNameDisplay.textContent = `File: ${result.filename}`;
            fileNameDisplay.classList.remove('hidden');

            // Enable the function input and load button
            document.getElementById('functionNameInput').disabled = false;
            updateLoadButtonState();

        } else {
            uploadStatus.textContent = `Error: ${result.detail}`;
            uploadStatus.className = 'text-sm text-red-600';
        }

    } catch (error) {
        uploadStatus.textContent = `Upload failed: ${error.message}`;
        uploadStatus.className = 'text-sm text-red-600';
    }
});

// Function to update load button state
function updateLoadButtonState() {
    const functionInput = document.getElementById('functionNameInput');
    const loadBtn = document.getElementById('loadGraphBtn');

    loadBtn.disabled = !currentTempPath || !functionInput.value.trim();
}

// Function input change handler
document.getElementById('functionNameInput').addEventListener('input', updateLoadButtonState);

// Load Graph button event handler
document.getElementById('loadGraphBtn').addEventListener('click', async () => {
    const functionInput = document.getElementById('functionNameInput');
    const functionName = functionInput.value.trim();

    if (!currentTempPath) {
        alert('Please upload a Python file first.');
        return;
    }

    if (!functionName) {
        alert('Please enter a function name.');
        return;
    }

    // Show loading spinner
    const spinner = document.getElementById('loading-spinner');
    spinner.classList.remove('hidden');

    // Hide any previous error messages
    const errorElement = document.getElementById('graph-error');
    if (errorElement) {
        errorElement.style.display = 'none';
    }

    try {
        await fetchAndRenderGraph(currentTempPath, functionName);
    } catch (error) {
        // Error handling is done in fetchAndRenderGraph
        console.error('Failed to load graph:', error);
    } finally {
        // Hide loading spinner
        spinner.classList.add('hidden');
    }
});

// Process map nodes data to prepare for visual grouping
function processCompositeNodeData(graphData) {
    // Create a mapping of composite root nodes (map, parallel, and conditional)
    const mapRoots = new Map();
    const parallelRoots = new Map();
    const conditionalRoots = new Map();

    // Track next nodes for parallel nodes to exclude them from bounding boxes
    const parallelNextNodes = new Map();

    // Find all composite root nodes and organize their related branch nodes
    graphData.nodes.forEach(node => {
        console.log("Examining node for composite properties:", node);

        // Identify map nodes either by label or metadata.node_type
        const isMapNode = node.label === "map" || (node.metadata && node.metadata.node_type === "map");
        const isParallelNode = node.label === "parallel" || (node.metadata && node.metadata.node_type === "parallel");
        const isConditionalNode = node.label === "conditional" || (node.metadata && node.metadata.node_type === "conditional");

        // Check if node has map-related metadata
        if (isMapNode) {
            console.log("Found map node:", node);

            // Get iteration metadata from node's metadata if available
            const iterate_on = node.metadata?.iterate_on || "items";
            const iterate_as = node.metadata?.iterate_as || "item";

            console.log(`Map node ${node.id} iterates "${iterate_as}" over "${iterate_on}"`);

            mapRoots.set(node.id, {
                mapNode: node,
                branches: new Map(), // Changed from branchNodes: []
                iterate_on: iterate_on,
                iterate_as: iterate_as
            });
        }

        // Check if node has parallel-related metadata
        if (isParallelNode) {
            console.log("Found parallel node:", node);

            parallelRoots.set(node.id, {
                parallelNode: node,
                branches: new Map() // Changed from branchNodes: []
            });

            // Find next node if there is a link from this parallel node
            if (graphData.links) {
                const nextLinks = graphData.links.filter(link => {
                    if (link.source !== node.id) {
                        return false;
                    }
                    // A "next" node is one that is NOT a direct child branch of the parallel node.
                    // Child branches have `belongs_to_node` pointing to the parallel node's ID.
                    const targetNode = graphData.nodes.find(n => n.id === link.target);
                    return !(targetNode && targetNode.metadata && targetNode.metadata.belongs_to_node === node.id);
                });

                if (nextLinks.length > 0) {
                    // Store the next node id
                    parallelNextNodes.set(node.id, nextLinks.map(link => link.target));
                    console.log(`Parallel node ${node.id} has next nodes:`, nextLinks.map(link => link.target));
                }
            }
        }

        // Check if node has conditional-related metadata
        if (isConditionalNode) {
            console.log("Found conditional node:", node);

            conditionalRoots.set(node.id, {
                conditionalNode: node,
                branches: new Map()
            });
        }

        // If this node belongs to a map or parallel branch, record it
        if (node.metadata && node.metadata.belongs_to_node) {
            const parentId = node.metadata.belongs_to_node;

            // Find the parent node to determine if it's a map or parallel
            const parentNode = graphData.nodes.find(n => n.id === parentId);

            if (parentNode) {
                const isParentParallel = parentNode.label === "parallel" ||
                                      (parentNode.metadata && parentNode.metadata.node_type === "parallel");
                const isParentConditional = parentNode.label === "conditional" ||
                                          (parentNode.metadata && parentNode.metadata.node_type === "conditional");

                if (isParentParallel) {
                    // This is a parallel branch node
                    console.log(`Node ${node.id} belongs to parallel node: ${parentId}`);

                    if (!parallelRoots.has(parentId)) {
                        parallelRoots.set(parentId, {
                            parallelNode: parentNode,
                            branches: new Map() // Changed
                        });
                    }

                    // NEW: group by branch
                    let branchName = 'branch';
                    if (node.id.startsWith(parentId + '.')) {
                        branchName = node.id.substring(parentId.length + 1).split('.')[0];
                    }
                    const root = parallelRoots.get(parentId);
                    if (!root.branches.has(branchName)) {
                        root.branches.set(branchName, []);
                    }
                    root.branches.get(branchName).push(node);

                } else if (isParentConditional) {
                    // This is a conditional branch node
                    console.log(`Node ${node.id} belongs to conditional node: ${parentId}`);

                    if (!conditionalRoots.has(parentId)) {
                        conditionalRoots.set(parentId, {
                            conditionalNode: parentNode,
                            branches: new Map()
                        });
                    }

                    // Group by branch (success/failure typically)
                    let branchName = 'branch';
                    if (node.id.startsWith(parentId + '.')) {
                        branchName = node.id.substring(parentId.length + 1).split('.')[0];
                    }
                    const root = conditionalRoots.get(parentId);
                    if (!root.branches.has(branchName)) {
                        root.branches.set(branchName, []);
                    }
                    root.branches.get(branchName).push(node);

                } else {
                    // This is a map branch node
                    console.log(`Node ${node.id} belongs to map: ${parentId}`);

                    if (!mapRoots.has(parentId)) {
                        mapRoots.set(parentId, {
                            mapNode: parentNode,
                            branches: new Map(), // Changed
                            iterate_on: parentNode.metadata?.iterate_on || "unknown",
                            iterate_as: parentNode.metadata?.iterate_as || "item"
                        });
                    }

                    // NEW: group by branch
                    let branchName = 'branch';
                    if (node.id.startsWith(parentId + '.')) {
                        branchName = node.id.substring(parentId.length + 1).split('.')[0];
                    }
                    const root = mapRoots.get(parentId);
                    if (!root.branches.has(branchName)) {
                        root.branches.set(branchName, []);
                    }
                    root.branches.get(branchName).push(node);
                }
            } else {
                // Fallback if parent node not found
                console.log(`Parent node ${parentId} not found for node ${node.id}`);
            }
        }
    });

    // Initialize mapGroups array in graphData
    graphData.mapGroups = [];

    // Process map roots
    mapRoots.forEach((value, key) => {
        console.log(`Processing map root: ${key}`);
        value.branches.forEach((branchNodes, branchName) => {
            graphData.mapGroups.push({
                id: `${key}-${branchName}`,
                type: 'map',
                nodes: branchNodes,
                rootNode: value.mapNode,
                branchName: branchName,
                iterate_on: value.iterate_on,
                iterate_as: value.iterate_as
            });
        });
    });

    // Process parallel roots
    parallelRoots.forEach((value, key) => {
        console.log(`Processing parallel root: ${key}`);

        const nextNodeIds = parallelNextNodes.get(key) || [];

        value.branches.forEach((branchNodes, branchName) => {
            graphData.mapGroups.push({
                id: `${key}-${branchName}`,
                type: 'parallel',
                nodes: branchNodes,
                rootNode: value.parallelNode,
                branchName: branchName,
                nextNodeIds: nextNodeIds
            });
        });
    });

    // Process conditional roots
    conditionalRoots.forEach((value, key) => {
        console.log(`Processing conditional root: ${key}`);

        value.branches.forEach((branchNodes, branchName) => {
            graphData.mapGroups.push({
                id: `${key}-${branchName}`,
                type: 'conditional',
                nodes: branchNodes,
                rootNode: value.conditionalNode,
                branchName: branchName
            });
        });
    });

    console.log("Final mapGroups:", graphData.mapGroups);
}

// File input change handler to show selected file name
document.getElementById('fileInput').addEventListener('change', function(event) {
    const fileNameDisplay = document.getElementById('file-name-display');
    const file = event.target.files[0];

    if (file) {
        fileNameDisplay.textContent = `Selected: ${file.name}`;
        fileNameDisplay.classList.remove('hidden');
    } else {
        fileNameDisplay.textContent = 'No file selected';
        fileNameDisplay.classList.add('hidden');
    }
});

// Dark mode toggle functionality
function initTheme() {
    // Check for saved theme preference or default to light mode
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

    if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
        document.documentElement.classList.add('dark');
        updateThemeToggle(true);
    } else {
        document.documentElement.classList.remove('dark');
        updateThemeToggle(false);
    }

    // Update SVG text colors to match theme
    updateSVGTextColors();
}

function updateThemeToggle(isDark) {
    const toggle = document.getElementById('theme-toggle');
    const lightIcon = document.getElementById('light-icon');
    const darkIcon = document.getElementById('dark-icon');

    if (isDark) {
        toggle.setAttribute('aria-checked', 'true');
        lightIcon.style.opacity = '0';
        darkIcon.style.opacity = '1';
    } else {
        toggle.setAttribute('aria-checked', 'false');
        lightIcon.style.opacity = '1';
        darkIcon.style.opacity = '0';
    }
}

function toggleTheme() {
    const isDark = document.documentElement.classList.contains('dark');

    if (isDark) {
        document.documentElement.classList.remove('dark');
        localStorage.setItem('theme', 'light');
        updateThemeToggle(false);
    } else {
        document.documentElement.classList.add('dark');
        localStorage.setItem('theme', 'dark');
        updateThemeToggle(true);
    }

    // Update SVG text colors when theme changes
    updateSVGTextColors();
}

function updateSVGTextColors() {
    const isDark = document.documentElement.classList.contains('dark');
    const textColor = isDark ? "#d1d5db" : "#4b5563";

    // Update zoom control text colors
    svg.selectAll(".zoom-controls text").attr("fill", textColor);

    // Update node text colors
    svg.selectAll(".nodes text").attr("fill", textColor);
}

// Initialize theme on page load
initTheme();

// Add event listener for theme toggle
document.getElementById('theme-toggle').addEventListener('click', toggleTheme);

// Listen for system theme changes
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
    if (!localStorage.getItem('theme')) {
        if (e.matches) {
            document.documentElement.classList.add('dark');
            updateThemeToggle(true);
        } else {
            document.documentElement.classList.remove('dark');
            updateThemeToggle(false);
        }
        // Update SVG text colors when system theme changes
        updateSVGTextColors();
    }
});
