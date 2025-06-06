import {
  Avatar,
  Card,
  Col,
  ColProps,
  Row,
  Space,
  theme,
  Typography,
} from 'antd';
import { ReactNode, useEffect, useState } from 'react';

import classNames from 'classnames';
import styles from './index.less';

type OptionType = {
  icon?: ReactNode;
  label: ReactNode;
  value: string;
  description?: string;
  disabled?: boolean;
};
type PropsType = {
  value?: string;
  defaultValue?: string;
  onChange?: (str: string, record: any) => void;
  options?: OptionType[];
  disabled?: boolean;
  layout?: ColProps;
  gutter?: [number, number];
};

export const CheckCard = ({
  value,
  defaultValue,
  onChange = () => {},
  options = [],
  disabled,
  layout = {
    xs: 24,
    sm: 24,
    md: 12,
    lg: 6,
    xl: 6,
    xxl: 6,
  },
  gutter = [16, 16],
}: PropsType) => {
  const { token } = theme.useToken();
  const [currentValue, setCurrentValue] = useState<string | undefined>(
    value || defaultValue,
  );
  // const { token } = theme.useToken();
  const onClick = (record: OptionType) => {
    if (disabled || record.disabled) return;
    setCurrentValue(record.value);
    onChange(record.value, record);
  };

  useEffect(() => {
    setCurrentValue(value);
  }, [value]);

  return (
    <Row gutter={gutter}>
      {options.map((option, key) => (
        <Col key={key} {...layout}>
          <Card
            className={classNames({
              [styles.item]: true,
              [styles.selected]: currentValue === option.value,
              [styles.disabled]: disabled || option.disabled,
            })}
            styles={{
              body: {
                padding: '8px 12px',
                borderRadius: token.borderRadius,
              },
            }}
            onClick={() => {
              onClick(option);
            }}
          >
            <Space className={styles.row}>
              <Space style={{ flex: 1 }}>
                {option.icon ? (
                  <Avatar shape="square" size={32} src={option.icon} />
                ) : null}
                <Space direction="vertical" style={{ flex: 1 }}>
                  <Typography.Text disabled={disabled || option.disabled}>
                    {option.label}
                  </Typography.Text>
                  {option.description ? (
                    <Typography.Text type="secondary" ellipsis>
                      {option.description}
                    </Typography.Text>
                  ) : null}
                </Space>
              </Space>
            </Space>
          </Card>
        </Col>
      ))}
    </Row>
  );
};
