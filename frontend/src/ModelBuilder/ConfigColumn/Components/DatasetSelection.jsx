function DatasetSelection({ selectedItem, dropdownOpen, toggleDropdown, items, handleItemClick }) {
  return (
    <div>
      <div className="dropdown" style={styles.dropdown}>
        <button
          className="dropdown-button"
          style={styles.dropdownButton}
          onClick={toggleDropdown}
        >
          {selectedItem ? `Selected: ${selectedItem}` : "Select Dataset"}
        </button>
        {dropdownOpen && (
          <div className="dropdown-content" style={styles.dropdownContent}>
            {items.map((item) => (
              <div
                key={item.value}
                style={styles.dropdownItem}
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
