
function Header(el)
  
  -- Add "center" to headers with class "title-slide-title"
  
  if el.classes:includes("title-slide-title") then
    if not el.classes:includes("center") then
      table.insert(el.classes, "center")
    end
  end
  
  -- Add "center" to headers with class "transition-slide"
  
  if el.classes:includes("transition-slide") then
    if not el.classes:includes("center") then
      table.insert(el.classes, "center")
    end
  end
  
  return el
end

