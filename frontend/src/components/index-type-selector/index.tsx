import {
  Tag,
  Typography,
  theme,
  Tooltip,
} from 'antd';
import { ReactNode, useEffect, useState } from 'react';
import { useIntl } from 'umi';
import classNames from 'classnames';
import styles from './index.less';

type IndexType = 'vector' | 'fulltext' | 'graph' | 'summary' | 'vision';

type IndexTypeOption = {
  key: IndexType;
  icon: string;
  required?: boolean;
  dependency?: string;
};

type PropsType = {
  value?: IndexType[];
  defaultValue?: IndexType[];
  onChange?: (selectedTypes: IndexType[]) => void;
  disabled?: boolean;
};

const INDEX_TYPE_OPTIONS: IndexTypeOption[] = [
  {
    key: 'vector',
    icon: 'VEC',
    required: true,
    dependency: 'embedding',
  },
  {
    key: 'fulltext',
    icon: 'TXT',
    required: true,
  },
  {
    key: 'graph',
    icon: 'GRH',
    dependency: 'completion',
  },
  {
    key: 'summary',
    icon: 'SUM',
    dependency: 'completion',
  },
  {
    key: 'vision',
    icon: 'VIS',
    dependency: 'completion',
  },
];



export const IndexTypeSelector = ({
  value,
  defaultValue = ['vector', 'fulltext', 'graph'],
  onChange = () => {},
  disabled,
}: PropsType) => {
  const { formatMessage } = useIntl();
  const { token } = theme.useToken();
  const [selectedTypes, setSelectedTypes] = useState<IndexType[]>(
    value || defaultValue,
  );

  const handleTypeToggle = (type: IndexType, option: IndexTypeOption) => {
    if (disabled) return;
    
    // Required types cannot be deselected
    if (option.required && selectedTypes.includes(type)) return;

    const newSelected = selectedTypes.includes(type)
      ? selectedTypes.filter((t) => t !== type)
      : [...selectedTypes, type];

    setSelectedTypes(newSelected);
    onChange(newSelected);
  };

  useEffect(() => {
    if (value) {
      setSelectedTypes(value);
    }
  }, [value]);

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <Typography.Title level={4} style={{ margin: 0 }}>
          {formatMessage({ id: 'collection.index_types' })}
        </Typography.Title>
        <Typography.Text type="secondary">
          {formatMessage({ id: 'collection.index_types.description' })}
        </Typography.Text>
      </div>

      <div className={styles.indexTypeList} style={{ marginTop: 20 }}>
        {INDEX_TYPE_OPTIONS.map((option, index) => {
          const isSelected = selectedTypes.includes(option.key);
          const isDisabled = disabled;

          const cannotDeselect = option.required && isSelected;
          
          const itemElement = (
            <div
              key={option.key}
              className={classNames(styles.indexTypeItem, {
                [styles.selected]: isSelected,
                [styles.disabled]: isDisabled,
                [styles.clickable]: !isDisabled && !cannotDeselect,
                [styles.required]: option.required,
                [styles.cannotDeselect]: cannotDeselect,
              })}
              onClick={() => handleTypeToggle(option.key, option)}
              style={{ marginBottom: index === INDEX_TYPE_OPTIONS.length - 1 ? 0 : 16 }}
            >
              <div className={styles.itemContent}>
                <div className={styles.iconWrapper}>
                  <span className={styles.iconText}>{option.icon}</span>
                </div>
                <div className={styles.content}>
                  <div className={styles.titleRow}>
                    <Typography.Text className={styles.title}>
                      {formatMessage({
                        id: `collection.index_type.${option.key}`,
                      })}
                    </Typography.Text>
                    {option.required && (
                      <Tag className={styles.requiredTag}>
                        {formatMessage({ id: 'collection.index_type.required' })}
                      </Tag>
                    )}
                  </div>
                  <Typography.Text className={styles.description}>
                    {formatMessage({
                      id: `collection.index_type.${option.key}.description`,
                    })}
                  </Typography.Text>
                  {option.dependency ? (
                    <div className={styles.dependency}>
                      <Typography.Text className={styles.dependencyText}>
                        {formatMessage({ id: 'collection.index_type.requires' })} {
                          option.dependency === 'completion' ? formatMessage({ id: 'collection.index_type.completion' }) : 
                          formatMessage({ id: 'collection.index_type.embedding' })
                        } {formatMessage({ id: 'collection.index_type.model' })}
                      </Typography.Text>
                    </div>
                  ) : (
                    <div className={styles.dependency} style={{ visibility: 'hidden' }}>
                      <Typography.Text className={styles.dependencyText}>
                        &nbsp;
                      </Typography.Text>
                    </div>
                  )}
                </div>
              </div>
            </div>
          );

          // Wrap with tooltip for required items that cannot be deselected
          return cannotDeselect ? (
            <Tooltip
              key={option.key}
              title={formatMessage({ id: 'collection.index_type.required.tooltip' })}
            >
              {itemElement}
            </Tooltip>
          ) : itemElement;
        })}
      </div>
    </div>
  );
};