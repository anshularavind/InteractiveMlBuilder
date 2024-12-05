import "../../ModelBuilder.css";

function LayerSelection({ selectedItem, dropdownOpen, toggleDropdown, handleItemClick, layers }) {
  let layerItems;
  if (layers.length > 0 && layers[layers.length - 1].type === "FcNN") {
    layerItems = [
      {value: "FcNN", label: "FcNN"},
    ];
  } else {
    layerItems = [
      {value: "FcNN", label: "FcNN"},
      {value: "Conv", label: "Conv"},
    ];
  }

  return (
    <div className="dropdown">
      <button className="dropdownButton" onClick={toggleDropdown} aria-expanded={dropdownOpen}>
        {selectedItem ? `Selected: ${selectedItem}` : "Select Model Type"}
      </button>
      {dropdownOpen && (
        <div className="dropdownContent" role="menu">
          {layerItems.map((item) => (
            <div
              key={item.value}
              className="dropdownItem"
              role="menuitem"
              onClick={() => handleItemClick(item)}
            >
              {item.label}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default LayerSelection;