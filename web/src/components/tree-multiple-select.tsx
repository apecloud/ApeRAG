import { cn } from '@/lib/utils';
import { CheckedState } from '@radix-ui/react-checkbox';
import { ChevronRight } from 'lucide-react';
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react';
import { Button } from './ui/button';
import { Checkbox } from './ui/checkbox';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from './ui/collapsible';
import { Label } from './ui/label';

export type TreeSelectItem = {
  id: string;
  name?: string;
  children?: TreeSelectItem[];
};

const TreeMultipleSelectContext = createContext<{
  values: string[];
  setValues: (v: string[]) => void;
  onValuesChange: (v: string[]) => void;
}>({
  values: [],
  setValues: () => {},
  onValuesChange: () => {},
});

const TreeSelectNode = ({
  item,
  level,
}: {
  item: TreeSelectItem;
  level: number;
}) => {
  const { values, setValues, onValuesChange } = useContext(
    TreeMultipleSelectContext,
  );

  const hasChildren = useMemo(
    () => item.children && item.children.length > 0,
    [item.children],
  );

  const handleCheckdChange = useCallback(
    (checked: CheckedState) => {
      let newValues = [...values];

      if (checked) {
        newValues.push(item.id);
      } else {
        newValues = newValues.filter((v) => v !== item.id);
      }

      const setChildrenChecked = (data?: TreeSelectItem[]) => {
        data?.forEach((child) => {
          if (checked) {
            newValues.push(child.id);
          } else {
            newValues = newValues.filter((v) => v !== child.id);
          }
          if (child.children?.length) {
            setChildrenChecked(child.children);
          }
        });
      };

      setChildrenChecked(item.children);
      setValues(newValues);
      onValuesChange(newValues);
    },
    [item.children, item.id, onValuesChange, setValues, values],
  );

  if (!hasChildren) {
    return (
      <Button
        variant="ghost"
        data-id={item.id}
        className={cn(`w-full cursor-default justify-start`)}
        style={{
          paddingLeft: level * 12 + 20,
        }}
        asChild
      >
        <Label className="flex-1">
          <Checkbox
            checked={values.includes(item.id)}
            onCheckedChange={handleCheckdChange}
          />
          {item.name}
        </Label>
      </Button>
    );
  }

  return (
    <Collapsible defaultOpen>
      <Button
        data-id={item.id}
        variant="ghost"
        className={cn(`w-full cursor-default justify-start`)}
        asChild
        style={{
          paddingLeft: level * 12 + 8,
        }}
      >
        <div>
          <CollapsibleTrigger asChild className="group/collapsible">
            <span className="cursor-pointer">
              <ChevronRight className="transition-transform duration-200 group-data-[state=open]/collapsible:rotate-90" />
            </span>
          </CollapsibleTrigger>
          <Label className="flex-1">
            <Checkbox
              checked={values.includes(item.id)}
              onCheckedChange={handleCheckdChange}
            />
            {item.name}
          </Label>
        </div>
      </Button>
      <CollapsibleContent>
        <TreeSelectNodes items={item.children} level={level + 1} />
      </CollapsibleContent>
    </Collapsible>
  );
};

const TreeSelectNodes = ({
  items = [],
  level = 0,
}: {
  items?: TreeSelectItem[];
  level?: number;
}) => {
  return items.map((item, i) => (
    <TreeSelectNode key={i} item={item} level={level} />
  ));
};

export type TreeSelectProps = {
  values?: string[];
  options?: TreeSelectItem[];
  placeholder?: string;
  onValuesChange?: (v: string[]) => void;
};

export const TreeMultipleSelect = ({
  values: initValues = [],
  options = [],
  onValuesChange = () => {},
  ...props
}: TreeSelectProps & React.ComponentProps<'div'>) => {
  const [values, setValues] = useState<string[]>(initValues || []);

  useEffect(() => {
    setValues(initValues);
  }, [initValues]);

  return (
    <TreeMultipleSelectContext.Provider
      value={{
        values,
        setValues,
        onValuesChange,
      }}
    >
      <div {...props}>
        <TreeSelectNodes items={options} />
      </div>
    </TreeMultipleSelectContext.Provider>
  );
};
