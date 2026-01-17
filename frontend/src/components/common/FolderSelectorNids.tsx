import React from 'react';
import { Button } from '@/components/ui/button';
import { Check, ChevronsUpDown, X } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { FolderSelectorProps } from '@/types/nids';

const FolderSelector: React.FC<FolderSelectorProps> = ({
  availableFolders,
  selectedFolders,
  onChange,
  loading,
  label = "Select folders",
  placeholder = "Search folders...",
}) => {
  const [open, setOpen] = React.useState(false);

  const handleSelect = (folder: string) => {
    const updatedSelection = selectedFolders.includes(folder)
      ? selectedFolders.filter(item => item !== folder)
      : [...selectedFolders, folder];
    
    onChange(updatedSelection);
    // Keep the popover open for multiple selections
  };

  const handleRemove = (folder: string, e: React.MouseEvent<HTMLButtonElement, MouseEvent>) => {
    e.stopPropagation(); // Prevent opening the popover
    const updatedSelection = selectedFolders.filter(item => item !== folder);
    onChange(updatedSelection);
  };

  return (
    <div className="w-full space-y-2">
      <label className="text-sm font-medium">{label}</label>
      
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className={`w-full justify-between h-auto min-h-10 ${
              selectedFolders.length > 0 ? "h-auto flex-wrap" : ""
            }`}
            disabled={loading || availableFolders.length === 0}
          >
            <div className="flex flex-wrap gap-1 py-1">
              {selectedFolders.length === 0 ? (
                <span className="text-muted-foreground">
                  {availableFolders.length === 0 
                    ? "No folders available" 
                    : "Select folders..."}
                </span>
              ) : (
                selectedFolders.map(folder => (
                  <Badge 
                    key={folder} 
                    variant="secondary" 
                    className="mr-1 mb-1 py-1 px-2"
                  >
                    {folder}
                    <button
                      className="ml-1 rounded-full outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
                      onMouseDown={e => e.preventDefault()}
                      onClick={e => handleRemove(folder, e)}
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </Badge>
                ))
              )}
            </div>
            <ChevronsUpDown className="h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-full p-0" align="start">
          <Command>
            <CommandInput placeholder={placeholder} />
            <CommandList>
              <CommandEmpty>No folders found</CommandEmpty>
              <CommandGroup>
                {availableFolders.map(folder => (
                  <CommandItem
                    key={folder}
                    value={folder}
                    onSelect={() => handleSelect(folder)}
                  >
                    <Check
                      className={`mr-2 h-4 w-4 ${
                        selectedFolders.includes(folder) 
                          ? "opacity-100" 
                          : "opacity-0"
                      }`}
                    />
                    {folder}
                  </CommandItem>
                ))}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
      
      {selectedFolders.length > 0 && (
        <div className="flex justify-between items-center">
          <p className="text-xs text-muted-foreground">
            {selectedFolders.length} folder{selectedFolders.length > 1 ? 's' : ''} selected
          </p>
          <Button 
            variant="ghost" 
            className="h-auto p-0 text-xs text-muted-foreground"
            onClick={() => onChange([])}
          >
            Clear all
          </Button>
        </div>
      )}
    </div>
  );
};

export default FolderSelector;