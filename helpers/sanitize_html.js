function cleanHTML(html_string,) {
    const tags_to_remove = ['script', 'style', 'link', 'meta', 'noscript', 'iframe', 'svg', 'code', 'noscript', 'i', 'img'];
    const query_selectors_to_remove = ["#ultimate_extension_div", '#toggle_ultimext'];
    const to_remove_if_empty = ['div', 'span', 'li', 'p', 'td', 'th', 'tr', 'table', 'a', 'button', 'input'];
    const tags_to_keep = new Set(['table', 'tr', 'th', 'td', 'thead', 'li', 'p']);

    // Create a DOM parser
    const parser = new DOMParser();
    const doc = parser.parseFromString(html_string, 'text/html');

    // Remove comments
    const commentIterator = doc.createNodeIterator(
        doc.documentElement,
        NodeFilter.SHOW_COMMENT,
        null,
        false
    );
    let comment;
    while (comment = commentIterator.nextNode()) {
        comment.remove();
    }

    tags_to_remove.forEach(tag => {
        for (const element of [...doc.getElementsByTagName(tag)]) {
            element.remove();
        }
    });

    for (const qs of query_selectors_to_remove) {
        for (const element of [...doc.querySelectorAll(qs)]) {
            element.remove();
        }
    }

    // Function to clean attributes
    function cleanAttributes(node) {
        if (!node.attributes) {
            return;
        }
        console.log(node);
        for (let i = node.attributes.length - 1; i >= 0; i--) {
            node.removeAttribute(node.attributes[i].name);
        }
    }

    function onlyChildPolicy(node) {
        cleanAttributes(node);
        const filtered_out_nodes = [...node.childNodes].filter(x => x.data?.trim() === "");

        for (const child_node of filtered_out_nodes) {
            node.removeChild(child_node);
        }

        if (node.childNodes.length === 1) {
            let child_node = node.childNodes[0];
            if (tags_to_keep.has(node.tagName.toLowerCase())) {
                node.replaceChild(onlyChildPolicy(child_node), child_node);
                return node;
            }
            if (child_node.nodeType === Node.TEXT_NODE) {
                child_node.data = " " + child_node.data;
                return child_node;
            }
            return onlyChildPolicy(child_node);
        }

        for (let child_node of node.childNodes) {
            new_child = onlyChildPolicy(child_node);
            try {
                node.replaceChild(new_child, child_node);
            } catch(e) {
                // Nothing, I don't care.
            }
        }

        return node;
    }

    // Start simplifying from the body
    onlyChildPolicy(doc.body);

    const sorted_elements = [...doc.body.querySelectorAll(to_remove_if_empty.join(','))].sort((x, y) => x.innerHTML.trim().length - y.innerHTML.trim() ? 1 : -1);

    for (const element of sorted_elements) {
        if (element.innerHTML.trim().length === 0) {
            element.remove();
        }
    }
    // Return the cleaned HTML
    if (!doc.body.firstElementChild) {
        return doc.body.outerHTML;
    }
    doc.body.firstElementChild.classList.add('context_element');
    return normalizeWhitespace(doc.body.innerHTML);
}


function normalizeWhitespace(str) {
    let oldStr;
    do {
        oldStr = str;
        // Replace 2+ line breaks with single line break
        str = str.replace(/\n\s*\n/g, '\n');
        // Replace 2+ spaces with single space
        str = str.replace(/[ \t]+/g, ' ');
    } while (oldStr !== str);

    return str;
}


function cleanHTML2(html_string) {
    let last_length = Infinity;
    let html = html_string;
    while (html.length < last_length) {
        // console.log(html.length)
        last_length = html.length
        html = cleanHTML(html)
    }

    return html;
}