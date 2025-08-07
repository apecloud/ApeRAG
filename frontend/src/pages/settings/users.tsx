import { Invitation, User } from '@/api';
import { PageContainer, PageHeader, RefreshButton } from '@/components';
import { DATETIME_FORMAT } from '@/constants';
import { api } from '@/services';
import { CheckOutlined, CloseOutlined, MoreOutlined } from '@ant-design/icons';
import { Button, Dropdown, Table, TableProps, Tabs, Typography } from 'antd';
import moment from 'moment';
import { useCallback, useEffect, useState } from 'react';

import { FormattedMessage, useIntl } from 'umi';

export default () => {
  const { formatMessage } = useIntl();
  const [users, setUsers] = useState<User[]>();
  const [usersLoading, setUsersLoading] = useState<boolean>();

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [invitations, setInvitations] = useState<Invitation[]>();
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [invitationsLoading, setInvitationsLoading] = useState<boolean>();

  const getUsers = useCallback(async () => {
    setUsersLoading(true);
    const res = await api.usersGet();
    setUsersLoading(false);
    setUsers(res.data.items);
  }, []);

  const getInvitations = useCallback(async () => {
    setInvitationsLoading(true);

    const res = await api.invitationsGet();
    setInvitationsLoading(false);
    setInvitations(res.data.items);
  }, []);

  const usersColumns: TableProps<User>['columns'] = [
    {
      title: formatMessage({ id: 'user.username' }),
      dataIndex: 'username',
      render: (value) => {
        return value ? (
          value
        ) : (
          <Typography.Text type="secondary" style={{ fontStyle: 'italic' }}>
            {formatMessage({ id: 'user.username_placeholder' })}
          </Typography.Text>
        );
      },
    },
    {
      title: formatMessage({ id: 'user.email' }),
      dataIndex: 'email',
      render: (value) => {
        return value ? (
          value
        ) : (
          <Typography.Text type="secondary" style={{ fontStyle: 'italic' }}>
            {formatMessage({ id: 'user.email_placeholder' })}
          </Typography.Text>
        );
      },
    },
    {
      title: formatMessage({ id: 'user.role' }),
      dataIndex: 'role',
    },
    {
      title: formatMessage({ id: 'user.status' }),
      dataIndex: 'is_active',
      render: (value) => {
        return value ? (
          <Typography.Text type="success">
            <CheckOutlined />
          </Typography.Text>
        ) : (
          <Typography.Text type="danger">
            <CloseOutlined />
          </Typography.Text>
        );
      },
    },
    {
      title: formatMessage({ id: 'user.registrationSource' }),
      dataIndex: 'registration_source',
      width: 120,
      render: (value, record) => {
        const sourceMap: Record<string, { text: string; color: string }> = {
          local: { text: formatMessage({ id: 'user.source.local' }), color: 'default' },
          google: { text: 'Google', color: 'blue' },
          github: { text: 'GitHub', color: 'purple' },
        };
        
        if (!value) {
          return (
            <Typography.Text type="secondary" style={{ fontStyle: 'italic' }}>
              {formatMessage({ id: 'user.source.local' })}
            </Typography.Text>
          );
        }
        
        const source = sourceMap[value] || { text: value, color: 'default' };
        
        // For GitHub users, make it clickable to jump to GitHub profile
        if (value === 'github' && record.username) {
          return (
            <Typography.Link
              href={`https://github.com/${record.username}`}
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: source.color === 'purple' ? '#722ed1' : undefined }}
            >
              {source.text}
            </Typography.Link>
          );
        }
        
        return (
          <Typography.Text type={source.color as any}>
            {source.text}
          </Typography.Text>
        );
      },
    },
    {
      title: formatMessage({ id: 'text.createdAt' }),
      dataIndex: 'date_joined',
      width: 180,
      render: (value) => moment(value).format(DATETIME_FORMAT),
    },
    {
      title: formatMessage({ id: 'action.name' }),
      width: 80,
      render: () => {
        return (
          <Dropdown
            trigger={['click']}
            menu={{ items: [] }}
            overlayStyle={{ width: 160 }}
          >
            <Button type="text" icon={<MoreOutlined />} />
          </Dropdown>
        );
      },
    },
  ];

  useEffect(() => {
    getUsers();
    getInvitations();
  }, []);

  return (
    <PageContainer>
      <PageHeader
        title={formatMessage({ id: 'users.management' })}
        description={formatMessage({ id: 'users.management_tips' })}
      >
        <Button type="primary" disabled>
          <FormattedMessage id="users.invite" />
        </Button>
        <RefreshButton loading={usersLoading} onClick={() => getUsers()} />
      </PageHeader>

      <Tabs
        items={[
          {
            label: formatMessage({ id: 'users.all' }),
            key: 'users',
            children: (
              <Table
                rowKey="id"
                bordered
                columns={usersColumns}
                dataSource={users}
                loading={usersLoading}
              />
            ),
          },
          {
            label: formatMessage({ id: 'users.invitations' }),
            key: 'invitations',
            children: '',
            disabled: true,
          },
        ]}
      />
    </PageContainer>
  );
};
