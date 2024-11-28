import "../../ModelBuilder.css";

function LayerSelection({ selectedItem, dropdownOpen, toggleDropdown, layerItems, handleItemClick }) {
  return (
    <div>
      <div className="dropdown">
        <button
          className="dropdownButton"
          onClick={toggleDropdown}
        >
          {selectedItem ? `Selected: ${selectedItem}` : "Select Model Type"}
        </button>
        {dropdownOpen && (
          <div className="dropdownContent">
            {layerItems.map((item) => (
              <div
                key={item.value}
                className="dropdownItem"
                onClick={() => handleItemClick(item)}
              >
                {item.label}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default LayerSelection;