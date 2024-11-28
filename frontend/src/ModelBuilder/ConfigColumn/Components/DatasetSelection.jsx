import "../../ModelBuilder.css";

function DatasetSelection({ selectedItem, dropdownOpen, toggleDropdown, datasetItems, handleItemClick }) {
  return (
    <div>
      <div className="dropdownDatasets">
        <button
          className="dropdownButton"
          onClick={toggleDropdown}
        >
          {selectedItem ? `Selected: ${selectedItem}` : "Select Dataset"}
        </button>
        {dropdownOpen && (
          <div className="dropdownContent">
            {datasetItems.map((item) => (
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