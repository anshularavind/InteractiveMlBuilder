import "../../ModelBuilder.css";

function DatasetSelection({ selectedItem, dropdownOpen, toggleDropdown, items, handleItemClick }) {
  return (
    <div>
      <div className="dropdown">
        <button
          className="dropdownButton"
          onClick={toggleDropdown}
        >
          {selectedItem ? `Selected: ${selectedItem}` : "Select Dataset"}
        </button>
        {dropdownOpen && (
          <div className="dropdownContent">
            {items.map((item) => (
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

export default DatasetSelection;