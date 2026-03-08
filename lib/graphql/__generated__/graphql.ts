export type Maybe<T> = T | null;
export type InputMaybe<T> = Maybe<T>;
export type Exact<T extends { [key: string]: unknown }> = { [K in keyof T]: T[K] };
export type MakeOptional<T, K extends keyof T> = Omit<T, K> & { [SubKey in K]?: Maybe<T[SubKey]> };
export type MakeMaybe<T, K extends keyof T> = Omit<T, K> & { [SubKey in K]: Maybe<T[SubKey]> };
export type MakeEmpty<T extends { [key: string]: unknown }, K extends keyof T> = { [_ in K]?: never };
export type Incremental<T> = T | { [P in keyof T]?: P extends ' $fragmentName' | '__typename' ? T[P] : never };
/** All built-in and custom scalars, mapped to their actual values */
export type Scalars = {
  ID: { input: string; output: string; }
  String: { input: string; output: string; }
  Boolean: { input: boolean; output: boolean; }
  Int: { input: number; output: number; }
  Float: { input: number; output: number; }
  DateTime: { input: unknown; output: unknown; }
  JSON: { input: unknown; output: unknown; }
};

export enum ActivityLevel {
  Active = 'active',
  Light = 'light',
  Moderate = 'moderate',
  Sedentary = 'sedentary',
  VeryActive = 'very_active'
}

export type AddUserItemInput = {
  content: Scalars['String']['input'];
  displayOrder?: InputMaybe<Scalars['Int']['input']>;
  promptKey?: InputMaybe<Scalars['String']['input']>;
  type: UserItemType;
};

export enum AttendeeStatus {
  Attended = 'attended',
  Going = 'going',
  Interested = 'interested'
}

export type Category = {
  __typename: 'Category';
  id: Scalars['String']['output'];
  /** Categories with similar embeddings. */
  recommendedCategories: Array<Scalars['String']['output']>;
  /** AI-recommended events for this category. */
  recommendedEvents: Array<Event>;
  /** AI-recommended spaces for this category. */
  recommendedSpaces: Array<Space>;
};


export type CategoryRecommendedCategoriesArgs = {
  limit?: InputMaybe<Scalars['Int']['input']>;
};


export type CategoryRecommendedEventsArgs = {
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
};


export type CategoryRecommendedSpacesArgs = {
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
};

export type Connection = {
  __typename: 'Connection';
  createdAt: Scalars['DateTime']['output'];
  id: Scalars['ID']['output'];
  initialMessage: Maybe<Scalars['String']['output']>;
  initiator: User;
  lastMessage: Maybe<Message>;
  lastMessageAt: Maybe<Scalars['DateTime']['output']>;
  /** Messages in this connection, newest first. */
  messages: Array<Message>;
  otherUser: User;
  recipient: User;
  status: ConnectionStatus;
  targetUserItem: UserItem;
  unreadCount: Maybe<Scalars['Int']['output']>;
  updatedAt: Scalars['DateTime']['output'];
};

export enum ConnectionStatus {
  Accepted = 'accepted',
  Declined = 'declined',
  Pending = 'pending'
}

export type Coordinates = {
  __typename: 'Coordinates';
  lat: Scalars['Float']['output'];
  lon: Scalars['Float']['output'];
};

export type CreateEventInput = {
  categories?: InputMaybe<Array<Scalars['String']['input']>>;
  cover: Scalars['String']['input'];
  currency?: InputMaybe<Scalars['String']['input']>;
  description?: InputMaybe<Scalars['String']['input']>;
  endsAt?: InputMaybe<Scalars['String']['input']>;
  images?: InputMaybe<Array<Scalars['String']['input']>>;
  lat?: InputMaybe<Scalars['Float']['input']>;
  location?: InputMaybe<Scalars['String']['input']>;
  lon?: InputMaybe<Scalars['Float']['input']>;
  maxAttendees?: InputMaybe<Scalars['Int']['input']>;
  price?: InputMaybe<Scalars['Int']['input']>;
  spaceId: Scalars['ID']['input'];
  startsAt: Scalars['String']['input'];
  title: Scalars['String']['input'];
};

export type CreateSpaceInput = {
  categories?: InputMaybe<Array<Scalars['String']['input']>>;
  cover: Scalars['String']['input'];
  description?: InputMaybe<Scalars['String']['input']>;
  images?: InputMaybe<Array<Scalars['String']['input']>>;
  joinPolicy?: InputMaybe<JoinPolicy>;
  name: Scalars['String']['input'];
  slug: Scalars['String']['input'];
  visibility?: InputMaybe<SpaceVisibility>;
};

export type CreateTierInput = {
  description?: InputMaybe<Scalars['String']['input']>;
  interval: TierInterval;
  name: Scalars['String']['input'];
  price: Scalars['Int']['input'];
};

export type DailyMatch = {
  __typename: 'DailyMatch';
  distanceKm: Maybe<Scalars['Float']['output']>;
  score: Scalars['Float']['output'];
  user: User;
};

export enum Drinking {
  Never = 'never',
  Regularly = 'regularly',
  Sometimes = 'sometimes'
}

export enum EducationLevel {
  Bachelor = 'bachelor',
  HighSchool = 'high_school',
  Master = 'master',
  MiddleSchool = 'middle_school',
  Other = 'other',
  Phd = 'phd',
  Vocational = 'vocational'
}

export enum Ethnicity {
  BlackAfrican = 'black_african',
  EastAsian = 'east_asian',
  HispanicLatino = 'hispanic_latino',
  Indigenous = 'indigenous',
  MiddleEastern = 'middle_eastern',
  Mixed = 'mixed',
  Other = 'other',
  PacificIslander = 'pacific_islander',
  SouthAsian = 'south_asian',
  WhiteCaucasian = 'white_caucasian'
}

export type Event = {
  __typename: 'Event';
  attendeeCount: Scalars['Int']['output'];
  attendees: Array<EventAttendee>;
  categories: Array<Scalars['String']['output']>;
  coordinates: Maybe<Coordinates>;
  cover: Scalars['String']['output'];
  createdAt: Scalars['DateTime']['output'];
  createdBy: User;
  currency: Maybe<Scalars['String']['output']>;
  description: Maybe<Scalars['String']['output']>;
  endsAt: Maybe<Scalars['DateTime']['output']>;
  id: Scalars['ID']['output'];
  images: Array<Scalars['String']['output']>;
  /** True when the event requires purchasing a ticket */
  isPaid: Scalars['Boolean']['output'];
  location: Maybe<Scalars['String']['output']>;
  maxAttendees: Maybe<Scalars['Int']['output']>;
  /** Status of the currently authenticated user for this event (null if not authenticated or not RSVP'd) */
  myAttendeeStatus: Maybe<AttendeeStatus>;
  /** Payment status for the currently authenticated user (null if free event or no purchase) */
  myPaymentStatus: Maybe<PaymentStatus>;
  price: Maybe<Scalars['Int']['output']>;
  /** Events with similar embeddings (AI-recommended). Excludes this event. */
  recommendedEvents: Array<Event>;
  /** The space this event belongs to */
  space: Maybe<Space>;
  spaceId: Scalars['ID']['output'];
  startsAt: Scalars['DateTime']['output'];
  title: Scalars['String']['output'];
  updatedAt: Scalars['DateTime']['output'];
};


export type EventRecommendedEventsArgs = {
  limit?: InputMaybe<Scalars['Int']['input']>;
};

export type EventAttendee = {
  __typename: 'EventAttendee';
  attendedAt: Maybe<Scalars['DateTime']['output']>;
  eventId: Scalars['ID']['output'];
  id: Scalars['ID']['output'];
  paymentStatus: Maybe<PaymentStatus>;
  registeredAt: Scalars['DateTime']['output'];
  status: AttendeeStatus;
  user: Maybe<User>;
  userId: Scalars['ID']['output'];
};

export enum Gender {
  Man = 'man',
  NonBinary = 'non_binary',
  Woman = 'woman'
}

export enum HasChildren {
  No = 'no',
  Yes = 'yes'
}

export enum JoinPolicy {
  Apply = 'apply',
  InviteOnly = 'invite_only',
  Open = 'open'
}

export type Member = {
  __typename: 'Member';
  currentPeriodEnd: Maybe<Scalars['DateTime']['output']>;
  id: Scalars['ID']['output'];
  joinedAt: Scalars['DateTime']['output'];
  role: MemberRole;
  status: MemberStatus;
  subscriptionId: Maybe<Scalars['String']['output']>;
  tier: Maybe<MembershipTier>;
  user: User;
};

export enum MemberRole {
  Admin = 'admin',
  Member = 'member',
  Owner = 'owner'
}

export enum MemberStatus {
  Active = 'active',
  Pending = 'pending',
  Suspended = 'suspended',
  WaitingPayment = 'waiting_payment'
}

export type MembershipTier = {
  __typename: 'MembershipTier';
  currency: Scalars['String']['output'];
  description: Maybe<Scalars['String']['output']>;
  id: Scalars['ID']['output'];
  interval: TierInterval;
  isActive: Scalars['Boolean']['output'];
  name: Scalars['String']['output'];
  price: Scalars['Int']['output'];
  spaceId: Scalars['ID']['output'];
};

export type Message = {
  __typename: 'Message';
  connectionId: Scalars['ID']['output'];
  content: Scalars['String']['output'];
  createdAt: Scalars['DateTime']['output'];
  id: Scalars['ID']['output'];
  readAt: Maybe<Scalars['DateTime']['output']>;
  sender: User;
};

export type Mutation = {
  __typename: 'Mutation';
  addUserItem: UserItem;
  approveMember: Member;
  archiveTier: Scalars['Boolean']['output'];
  /** Create a new category with ML embeddings (64d + 256d). Returns the category id. */
  createCategory: Scalars['String']['output'];
  /** Create a new event in a space. */
  createEvent: Event;
  createPost: Post;
  createSpace: Space;
  createTier: MembershipTier;
  deleteNotification: Scalars['Boolean']['output'];
  deletePost: Scalars['Boolean']['output'];
  deleteSpace: Scalars['Boolean']['output'];
  deleteUser: Scalars['Boolean']['output'];
  deleteUserItem: Scalars['Boolean']['output'];
  joinSpace: Member;
  leaveSpace: Scalars['Boolean']['output'];
  markAllNotificationsRead: Scalars['Boolean']['output'];
  /** Mark all messages in a connection as read. */
  markAsRead: Maybe<Scalars['Boolean']['output']>;
  /** Mark an event as completed. Attendees with status 'going' become 'attended'. */
  markEventCompleted: Event;
  markNotificationRead: Maybe<Notification>;
  removeMember: Scalars['Boolean']['output'];
  reorderUserItems: Array<UserItem>;
  /** Respond to an event (going, interested). */
  respondToEvent: EventAttendee;
  /** Accept or decline a connection request. */
  respondToRequest: Connection;
  /** Send a connection request to another user. */
  sendConnectionRequest: Connection;
  /** Send a message in an existing connection. */
  sendMessage: Message;
  /** Update an existing event. */
  updateEvent: Event;
  updateLocation: User;
  updateMemberRole: Member;
  updateSpace: Space;
  updateTier: MembershipTier;
  updateUser: Maybe<User>;
  updateUserItem: UserItem;
};


export type MutationAddUserItemArgs = {
  input: AddUserItemInput;
};


export type MutationApproveMemberArgs = {
  spaceId: Scalars['ID']['input'];
  userId: Scalars['ID']['input'];
};


export type MutationArchiveTierArgs = {
  id: Scalars['ID']['input'];
};


export type MutationCreateCategoryArgs = {
  name: Scalars['String']['input'];
};


export type MutationCreateEventArgs = {
  input: CreateEventInput;
};


export type MutationCreatePostArgs = {
  content: Scalars['String']['input'];
  mediaUrls?: InputMaybe<Array<Scalars['String']['input']>>;
  spaceId: Scalars['ID']['input'];
};


export type MutationCreateSpaceArgs = {
  input: CreateSpaceInput;
};


export type MutationCreateTierArgs = {
  input: CreateTierInput;
  spaceId: Scalars['ID']['input'];
};


export type MutationDeleteNotificationArgs = {
  id: Scalars['ID']['input'];
};


export type MutationDeletePostArgs = {
  postId: Scalars['ID']['input'];
};


export type MutationDeleteSpaceArgs = {
  id: Scalars['ID']['input'];
};


export type MutationDeleteUserArgs = {
  id: Scalars['ID']['input'];
};


export type MutationDeleteUserItemArgs = {
  itemId: Scalars['ID']['input'];
};


export type MutationJoinSpaceArgs = {
  spaceSlug: Scalars['String']['input'];
  tierId?: InputMaybe<Scalars['ID']['input']>;
};


export type MutationLeaveSpaceArgs = {
  spaceId: Scalars['ID']['input'];
};


export type MutationMarkAsReadArgs = {
  connectionId: Scalars['ID']['input'];
};


export type MutationMarkEventCompletedArgs = {
  eventId: Scalars['ID']['input'];
};


export type MutationMarkNotificationReadArgs = {
  id: Scalars['ID']['input'];
};


export type MutationRemoveMemberArgs = {
  spaceId: Scalars['ID']['input'];
  userId: Scalars['ID']['input'];
};


export type MutationReorderUserItemsArgs = {
  itemIds: Array<Scalars['ID']['input']>;
};


export type MutationRespondToEventArgs = {
  eventId: Scalars['ID']['input'];
  status: AttendeeStatus;
};


export type MutationRespondToRequestArgs = {
  accept: Scalars['Boolean']['input'];
  connectionId: Scalars['ID']['input'];
};


export type MutationSendConnectionRequestArgs = {
  initialMessage?: InputMaybe<Scalars['String']['input']>;
  recipientId: Scalars['ID']['input'];
  targetUserItemId: Scalars['ID']['input'];
};


export type MutationSendMessageArgs = {
  connectionId: Scalars['ID']['input'];
  content: Scalars['String']['input'];
};


export type MutationUpdateEventArgs = {
  id: Scalars['ID']['input'];
  input: UpdateEventInput;
};


export type MutationUpdateLocationArgs = {
  lat: Scalars['Float']['input'];
  location?: InputMaybe<Scalars['String']['input']>;
  lon: Scalars['Float']['input'];
};


export type MutationUpdateMemberRoleArgs = {
  role: MemberRole;
  spaceId: Scalars['ID']['input'];
  userId: Scalars['ID']['input'];
};


export type MutationUpdateSpaceArgs = {
  id: Scalars['ID']['input'];
  input: UpdateSpaceInput;
};


export type MutationUpdateTierArgs = {
  id: Scalars['ID']['input'];
  input: UpdateTierInput;
};


export type MutationUpdateUserArgs = {
  id: Scalars['ID']['input'];
  input: UpdateUserInput;
};


export type MutationUpdateUserItemArgs = {
  input: UpdateUserItemInput;
  itemId: Scalars['ID']['input'];
};

export type Notification = {
  __typename: 'Notification';
  createdAt: Scalars['DateTime']['output'];
  href: Maybe<Scalars['String']['output']>;
  id: Scalars['ID']['output'];
  image: Maybe<Scalars['String']['output']>;
  read: Scalars['Boolean']['output'];
  text: Scalars['String']['output'];
  type: NotificationType;
};

export enum NotificationType {
  EventReminder = 'event_reminder',
  EventRsvp = 'event_rsvp',
  Generic = 'generic',
  MatchMutual = 'match_mutual',
  NewMatch = 'new_match',
  NewMessage = 'new_message',
  SpaceJoined = 'space_joined'
}

export type NotificationsResult = {
  __typename: 'NotificationsResult';
  items: Array<Notification>;
  unreadCount: Scalars['Int']['output'];
};

export enum PaymentStatus {
  Paid = 'paid',
  Pending = 'pending',
  Refunded = 'refunded'
}

export type Post = {
  __typename: 'Post';
  author: User;
  content: Scalars['String']['output'];
  createdAt: Scalars['DateTime']['output'];
  id: Scalars['ID']['output'];
  mediaUrls: Maybe<Array<Scalars['String']['output']>>;
  space: Space;
};

export type Query = {
  __typename: 'Query';
  /** All available interest categories (unique, sorted). */
  categories: Array<Category>;
  /** Single category by slug id, null if not found. */
  category: Maybe<Category>;
  /** Check if a username is already taken. */
  checkUsername: Scalars['Boolean']['output'];
  event: Maybe<Event>;
  events: Array<Event>;
  me: Maybe<User>;
  mySpaces: Array<Space>;
  myUpcomingEvents: Array<Event>;
  space: Maybe<Space>;
  spaces: Array<Space>;
  user: Maybe<User>;
};


export type QueryCategoryArgs = {
  id: Scalars['String']['input'];
};


export type QueryCheckUsernameArgs = {
  username: Scalars['String']['input'];
};


export type QueryEventArgs = {
  id: Scalars['ID']['input'];
};


export type QueryEventsArgs = {
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
};


export type QuerySpaceArgs = {
  id?: InputMaybe<Scalars['ID']['input']>;
  slug?: InputMaybe<Scalars['String']['input']>;
};


export type QueryUserArgs = {
  username: Scalars['String']['input'];
};

export enum RelationshipStyle {
  EthicalNonMonogamous = 'ethical_non_monogamous',
  Monogamous = 'monogamous',
  Open = 'open',
  Other = 'other'
}

export enum Religion {
  Buddhist = 'buddhist',
  Christian = 'christian',
  Hindu = 'hindu',
  Jewish = 'jewish',
  Muslim = 'muslim',
  None = 'none',
  Other = 'other',
  Spiritual = 'spiritual'
}

export enum Smoking {
  Never = 'never',
  Regularly = 'regularly',
  Sometimes = 'sometimes'
}

export type Space = {
  __typename: 'Space';
  categories: Array<Scalars['String']['output']>;
  cover: Scalars['String']['output'];
  createdAt: Scalars['DateTime']['output'];
  description: Maybe<Scalars['String']['output']>;
  /** Events belonging to this space, ordered by startsAt. */
  events: Array<Event>;
  feed: Array<Post>;
  id: Scalars['ID']['output'];
  images: Array<Scalars['String']['output']>;
  joinPolicy: JoinPolicy;
  members: Array<Member>;
  membersCount: Maybe<Scalars['Int']['output']>;
  myMembership: Maybe<Member>;
  name: Scalars['String']['output'];
  /** Upcoming events from other spaces with similar embeddings (AI-recommended). */
  recommendedEvents: Array<Event>;
  slug: Scalars['String']['output'];
  stripeAccountEnabled: Scalars['Boolean']['output'];
  tiers: Array<MembershipTier>;
  visibility: SpaceVisibility;
};


export type SpaceEventsArgs = {
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
};


export type SpaceFeedArgs = {
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
};


export type SpaceMembersArgs = {
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
};


export type SpaceRecommendedEventsArgs = {
  limit?: InputMaybe<Scalars['Int']['input']>;
};

export enum SpaceType {
  Free = 'free',
  Tiered = 'tiered'
}

export enum SpaceVisibility {
  Hidden = 'hidden',
  Private = 'private',
  Public = 'public'
}

export enum TierInterval {
  Month = 'month',
  OneTime = 'one_time',
  Year = 'year'
}

export type UpdateEventInput = {
  categories?: InputMaybe<Array<Scalars['String']['input']>>;
  cover?: InputMaybe<Scalars['String']['input']>;
  currency?: InputMaybe<Scalars['String']['input']>;
  description?: InputMaybe<Scalars['String']['input']>;
  endsAt?: InputMaybe<Scalars['String']['input']>;
  images?: InputMaybe<Array<Scalars['String']['input']>>;
  lat?: InputMaybe<Scalars['Float']['input']>;
  location?: InputMaybe<Scalars['String']['input']>;
  lon?: InputMaybe<Scalars['Float']['input']>;
  maxAttendees?: InputMaybe<Scalars['Int']['input']>;
  price?: InputMaybe<Scalars['Int']['input']>;
  startsAt?: InputMaybe<Scalars['String']['input']>;
  title?: InputMaybe<Scalars['String']['input']>;
};

export type UpdateSpaceInput = {
  categories?: InputMaybe<Array<Scalars['String']['input']>>;
  cover?: InputMaybe<Scalars['String']['input']>;
  description?: InputMaybe<Scalars['String']['input']>;
  images?: InputMaybe<Array<Scalars['String']['input']>>;
  joinPolicy?: InputMaybe<JoinPolicy>;
  name?: InputMaybe<Scalars['String']['input']>;
  visibility?: InputMaybe<SpaceVisibility>;
};

export type UpdateTierInput = {
  description?: InputMaybe<Scalars['String']['input']>;
  interval?: InputMaybe<TierInterval>;
  isActive?: InputMaybe<Scalars['Boolean']['input']>;
  name?: InputMaybe<Scalars['String']['input']>;
  price?: InputMaybe<Scalars['Int']['input']>;
};

export type UpdateUserInput = {
  activityLevel?: InputMaybe<ActivityLevel>;
  birthdate?: InputMaybe<Scalars['String']['input']>;
  drinking?: InputMaybe<Drinking>;
  educationLevel?: InputMaybe<EducationLevel>;
  email?: InputMaybe<Scalars['String']['input']>;
  ethnicity?: InputMaybe<Ethnicity>;
  gender?: InputMaybe<Gender>;
  hasChildren?: InputMaybe<HasChildren>;
  heightCm?: InputMaybe<Scalars['Int']['input']>;
  jobTitle?: InputMaybe<Scalars['String']['input']>;
  languages?: InputMaybe<Array<Scalars['String']['input']>>;
  location?: InputMaybe<Scalars['String']['input']>;
  name?: InputMaybe<Scalars['String']['input']>;
  relationshipIntent?: InputMaybe<Array<Scalars['String']['input']>>;
  relationshipStyle?: InputMaybe<RelationshipStyle>;
  religion?: InputMaybe<Religion>;
  schoolName?: InputMaybe<Scalars['String']['input']>;
  sexualOrientation?: InputMaybe<Array<Scalars['String']['input']>>;
  smoking?: InputMaybe<Smoking>;
  username?: InputMaybe<Scalars['String']['input']>;
  wantsChildren?: InputMaybe<WantsChildren>;
};

export type UpdateUserItemInput = {
  content?: InputMaybe<Scalars['String']['input']>;
  promptKey?: InputMaybe<Scalars['String']['input']>;
};

/** User — demographic, auth, and location data */
export type User = {
  __typename: 'User';
  activityLevel: Maybe<ActivityLevel>;
  birthdate: Scalars['String']['output'];
  /** Single connection by ID (must belong to the authenticated user). */
  connection: Maybe<Connection>;
  /** Pending incoming connection requests. */
  connectionRequests: Array<Connection>;
  /** Accepted connections (chats) for the authenticated user. */
  connections: Array<Connection>;
  coordinates: Maybe<Coordinates>;
  createdAt: Scalars['DateTime']['output'];
  /**
   * Today's 8 pre-computed matches based on bidirectional embedding similarity.
   * Generates on-the-fly for the first request of the day, then cached in DB.
   * Only visible to the authenticated user on their own profile.
   */
  dailyMatches: Array<DailyMatch>;
  drinking: Maybe<Drinking>;
  educationLevel: Maybe<EducationLevel>;
  email: Scalars['String']['output'];
  ethnicity: Maybe<Ethnicity>;
  /** Posts from all spaces the user is an active member of. */
  feed: Array<Post>;
  gender: Maybe<Gender>;
  hasChildren: Maybe<HasChildren>;
  heightCm: Maybe<Scalars['Int']['output']>;
  id: Scalars['ID']['output'];
  image: Maybe<Scalars['String']['output']>;
  jobTitle: Maybe<Scalars['String']['output']>;
  languages: Array<Scalars['String']['output']>;
  location: Maybe<Scalars['String']['output']>;
  locationUpdatedAt: Maybe<Scalars['DateTime']['output']>;
  name: Scalars['String']['output'];
  notifications: NotificationsResult;
  /** Categories recommended based on embedding similarity. Only visible to own profile. */
  recommendedCategories: Array<Category>;
  /** Recommended events based on embedding similarity. Only visible to own profile. */
  recommendedEvents: Array<Event>;
  /** Recommended spaces based on embedding similarity. Only visible to own profile. */
  recommendedSpaces: Array<Space>;
  /** Users with similar embeddings. Only visible to own profile. */
  recommendedUsers: Array<User>;
  relationshipIntent: Array<Scalars['String']['output']>;
  relationshipStyle: Maybe<RelationshipStyle>;
  religion: Maybe<Religion>;
  schoolName: Maybe<Scalars['String']['output']>;
  sexualOrientation: Array<Scalars['String']['output']>;
  smoking: Maybe<Smoking>;
  updatedAt: Scalars['DateTime']['output'];
  userItems: Array<UserItem>;
  username: Maybe<Scalars['String']['output']>;
  wantsChildren: Maybe<WantsChildren>;
};


/** User — demographic, auth, and location data */
export type UserConnectionArgs = {
  id: Scalars['ID']['input'];
};


/** User — demographic, auth, and location data */
export type UserFeedArgs = {
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
};


/** User — demographic, auth, and location data */
export type UserNotificationsArgs = {
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
};


/** User — demographic, auth, and location data */
export type UserRecommendedCategoriesArgs = {
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
};


/** User — demographic, auth, and location data */
export type UserRecommendedEventsArgs = {
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
};


/** User — demographic, auth, and location data */
export type UserRecommendedSpacesArgs = {
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
};


/** User — demographic, auth, and location data */
export type UserRecommendedUsersArgs = {
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
};

/** A single item on a user's profile — either a photo or a prompt answer. */
export type UserItem = {
  __typename: 'UserItem';
  content: Scalars['String']['output'];
  createdAt: Scalars['DateTime']['output'];
  displayOrder: Scalars['Int']['output'];
  id: Scalars['ID']['output'];
  promptKey: Maybe<Scalars['String']['output']>;
  type: UserItemType;
  updatedAt: Scalars['DateTime']['output'];
  userId: Scalars['ID']['output'];
};

export enum UserItemType {
  Photo = 'photo',
  Prompt = 'prompt'
}

export enum WantsChildren {
  No = 'no',
  Open = 'open',
  Yes = 'yes'
}

export type GetCategoriesQueryVariables = Exact<{ [key: string]: never; }>;


export type GetCategoriesQuery = { categories: Array<{ __typename: 'Category', id: string }> };

export type GetCategoryQueryVariables = Exact<{
  id: Scalars['String']['input'];
  eventsLimit?: InputMaybe<Scalars['Int']['input']>;
  spacesLimit?: InputMaybe<Scalars['Int']['input']>;
}>;


export type GetCategoryQuery = { category: { __typename: 'Category', id: string, recommendedCategories: Array<string>, recommendedEvents: Array<{ __typename: 'Event', id: string, title: string, description: string | null, location: string | null, startsAt: unknown, endsAt: unknown | null, spaceId: string, price: number | null, currency: string | null, isPaid: boolean, attendeeCount: number, maxAttendees: number | null, categories: Array<string> }>, recommendedSpaces: Array<{ __typename: 'Space', id: string, name: string, slug: string, description: string | null, cover: string, categories: Array<string>, visibility: SpaceVisibility, joinPolicy: JoinPolicy, membersCount: number | null, stripeAccountEnabled: boolean, createdAt: unknown }> } | null };

export type GetRecommendedCategoriesQueryVariables = Exact<{
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
}>;


export type GetRecommendedCategoriesQuery = { me: { __typename: 'User', id: string, recommendedCategories: Array<{ __typename: 'Category', id: string }> } | null };

export type ConnectionFieldsFragment = { __typename: 'Connection', id: string, status: ConnectionStatus, initialMessage: string | null, lastMessageAt: unknown | null, createdAt: unknown, updatedAt: unknown, unreadCount: number | null, targetUserItem: { __typename: 'UserItem', id: string, content: string, type: UserItemType }, otherUser: { __typename: 'User', id: string, name: string, image: string | null }, lastMessage: { __typename: 'Message', content: string, createdAt: unknown } | null };

export type MessageFieldsFragment = { __typename: 'Message', id: string, connectionId: string, content: string, readAt: unknown | null, createdAt: unknown, sender: { __typename: 'User', id: string, name: string, image: string | null } };

export type GetConnectionsQueryVariables = Exact<{ [key: string]: never; }>;


export type GetConnectionsQuery = { me: { __typename: 'User', id: string, connections: Array<{ __typename: 'Connection', id: string, status: ConnectionStatus, initialMessage: string | null, lastMessageAt: unknown | null, createdAt: unknown, updatedAt: unknown, unreadCount: number | null, targetUserItem: { __typename: 'UserItem', id: string, content: string, type: UserItemType }, otherUser: { __typename: 'User', id: string, name: string, image: string | null }, lastMessage: { __typename: 'Message', content: string, createdAt: unknown } | null }> } | null };

export type GetConnectionRequestsQueryVariables = Exact<{ [key: string]: never; }>;


export type GetConnectionRequestsQuery = { me: { __typename: 'User', id: string, connectionRequests: Array<{ __typename: 'Connection', id: string, status: ConnectionStatus, initialMessage: string | null, lastMessageAt: unknown | null, createdAt: unknown, updatedAt: unknown, unreadCount: number | null, targetUserItem: { __typename: 'UserItem', id: string, content: string, type: UserItemType }, otherUser: { __typename: 'User', id: string, name: string, image: string | null }, lastMessage: { __typename: 'Message', content: string, createdAt: unknown } | null }> } | null };

export type GetRecentConnectionsQueryVariables = Exact<{ [key: string]: never; }>;


export type GetRecentConnectionsQuery = { me: { __typename: 'User', id: string, connections: Array<{ __typename: 'Connection', id: string, unreadCount: number | null, otherUser: { __typename: 'User', name: string } }> } | null };

export type GetMessagesQueryVariables = Exact<{
  connectionId: Scalars['ID']['input'];
}>;


export type GetMessagesQuery = { me: { __typename: 'User', id: string, connection: { __typename: 'Connection', id: string, status: ConnectionStatus, otherUser: { __typename: 'User', id: string, name: string, image: string | null }, messages: Array<{ __typename: 'Message', id: string, connectionId: string, content: string, readAt: unknown | null, createdAt: unknown, sender: { __typename: 'User', id: string, name: string, image: string | null } }> } | null } | null };

export type SendConnectionRequestMutationVariables = Exact<{
  recipientId: Scalars['ID']['input'];
  targetUserItemId: Scalars['ID']['input'];
  initialMessage?: InputMaybe<Scalars['String']['input']>;
}>;


export type SendConnectionRequestMutation = { sendConnectionRequest: { __typename: 'Connection', id: string, status: ConnectionStatus, initialMessage: string | null, lastMessageAt: unknown | null, createdAt: unknown, updatedAt: unknown, unreadCount: number | null, targetUserItem: { __typename: 'UserItem', id: string, content: string, type: UserItemType }, otherUser: { __typename: 'User', id: string, name: string, image: string | null }, lastMessage: { __typename: 'Message', content: string, createdAt: unknown } | null } };

export type RespondToRequestMutationVariables = Exact<{
  connectionId: Scalars['ID']['input'];
  accept: Scalars['Boolean']['input'];
}>;


export type RespondToRequestMutation = { respondToRequest: { __typename: 'Connection', id: string, status: ConnectionStatus, initialMessage: string | null, lastMessageAt: unknown | null, createdAt: unknown, updatedAt: unknown, unreadCount: number | null, targetUserItem: { __typename: 'UserItem', id: string, content: string, type: UserItemType }, otherUser: { __typename: 'User', id: string, name: string, image: string | null }, lastMessage: { __typename: 'Message', content: string, createdAt: unknown } | null } };

export type SendMessageMutationVariables = Exact<{
  connectionId: Scalars['ID']['input'];
  content: Scalars['String']['input'];
}>;


export type SendMessageMutation = { sendMessage: { __typename: 'Message', id: string, content: string, createdAt: unknown } };

export type MarkAsReadMutationVariables = Exact<{
  connectionId: Scalars['ID']['input'];
}>;


export type MarkAsReadMutation = { markAsRead: boolean | null };

export type EventCardFieldsFragment = { __typename: 'Event', id: string, title: string, description: string | null, location: string | null, cover: string, startsAt: unknown, endsAt: unknown | null, maxAttendees: number | null, attendeeCount: number, myAttendeeStatus: AttendeeStatus | null, categories: Array<string>, spaceId: string, price: number | null, isPaid: boolean };

export type SpaceEventsQueryVariables = Exact<{
  spaceId: Scalars['ID']['input'];
}>;


export type SpaceEventsQuery = { space: { __typename: 'Space', id: string, events: Array<{ __typename: 'Event', currency: string | null, id: string, title: string, description: string | null, location: string | null, cover: string, startsAt: unknown, endsAt: unknown | null, maxAttendees: number | null, attendeeCount: number, myAttendeeStatus: AttendeeStatus | null, categories: Array<string>, spaceId: string, price: number | null, isPaid: boolean, createdBy: { __typename: 'User', id: string, name: string } }> } | null };

export type GetEventQueryVariables = Exact<{
  id: Scalars['ID']['input'];
}>;


export type GetEventQuery = { event: { __typename: 'Event', id: string, title: string, description: string | null, location: string | null, startsAt: unknown, endsAt: unknown | null, cover: string, images: Array<string>, maxAttendees: number | null, attendeeCount: number, myAttendeeStatus: AttendeeStatus | null, myPaymentStatus: PaymentStatus | null, categories: Array<string>, spaceId: string, createdAt: unknown, price: number | null, currency: string | null, isPaid: boolean, coordinates: { __typename: 'Coordinates', lat: number, lon: number } | null, createdBy: { __typename: 'User', id: string, name: string, username: string | null }, space: { __typename: 'Space', id: string, name: string, slug: string, visibility: SpaceVisibility, stripeAccountEnabled: boolean, myMembership: { __typename: 'Member', role: MemberRole } | null } | null, attendees: Array<{ __typename: 'EventAttendee', id: string, userId: string, status: AttendeeStatus, registeredAt: unknown, paymentStatus: PaymentStatus | null, user: { __typename: 'User', id: string, name: string, username: string | null } | null }> } | null };

export type MyUpcomingEventsQueryVariables = Exact<{ [key: string]: never; }>;


export type MyUpcomingEventsQuery = { myUpcomingEvents: Array<{ __typename: 'Event', id: string, title: string, description: string | null, location: string | null, cover: string, startsAt: unknown, endsAt: unknown | null, maxAttendees: number | null, attendeeCount: number, myAttendeeStatus: AttendeeStatus | null, categories: Array<string>, spaceId: string, price: number | null, isPaid: boolean }> };

export type GetAllEventsQueryVariables = Exact<{
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
}>;


export type GetAllEventsQuery = { events: Array<{ __typename: 'Event', id: string, title: string, description: string | null, location: string | null, cover: string, startsAt: unknown, endsAt: unknown | null, maxAttendees: number | null, attendeeCount: number, myAttendeeStatus: AttendeeStatus | null, categories: Array<string>, spaceId: string, price: number | null, isPaid: boolean }> };

export type UpdateEventMutationVariables = Exact<{
  id: Scalars['ID']['input'];
  input: UpdateEventInput;
}>;


export type UpdateEventMutation = { updateEvent: { __typename: 'Event', id: string, title: string, description: string | null, location: string | null, startsAt: unknown, endsAt: unknown | null, maxAttendees: number | null, categories: Array<string>, price: number | null, currency: string | null } };

export type CreateEventMutationVariables = Exact<{
  input: CreateEventInput;
}>;


export type CreateEventMutation = { createEvent: { __typename: 'Event', id: string } };

export type RespondToEventMutationVariables = Exact<{
  eventId: Scalars['ID']['input'];
  status: AttendeeStatus;
}>;


export type RespondToEventMutation = { respondToEvent: { __typename: 'EventAttendee', id: string, status: AttendeeStatus } };

export type MarkEventCompletedMutationVariables = Exact<{
  eventId: Scalars['ID']['input'];
}>;


export type MarkEventCompletedMutation = { markEventCompleted: { __typename: 'Event', id: string } };

export type GetEventRecommendedEventsQueryVariables = Exact<{
  id: Scalars['ID']['input'];
  limit?: InputMaybe<Scalars['Int']['input']>;
}>;


export type GetEventRecommendedEventsQuery = { event: { __typename: 'Event', id: string, recommendedEvents: Array<{ __typename: 'Event', id: string, title: string, description: string | null, location: string | null, cover: string, startsAt: unknown, endsAt: unknown | null, maxAttendees: number | null, attendeeCount: number, myAttendeeStatus: AttendeeStatus | null, categories: Array<string>, spaceId: string, price: number | null, isPaid: boolean, space: { __typename: 'Space', id: string, name: string, slug: string } | null }> } | null };

export type GetDailyMatchesQueryVariables = Exact<{ [key: string]: never; }>;


export type GetDailyMatchesQuery = { me: { __typename: 'User', id: string, dailyMatches: Array<{ __typename: 'DailyMatch', score: number, distanceKm: number | null, user: { __typename: 'User', id: string, username: string | null, name: string, image: string | null, gender: Gender | null, birthdate: string, userItems: Array<{ __typename: 'UserItem', id: string, type: UserItemType, content: string, displayOrder: number }> } }> } | null };

export type UpdateMemberRoleMutationVariables = Exact<{
  spaceId: Scalars['ID']['input'];
  userId: Scalars['ID']['input'];
  role: MemberRole;
}>;


export type UpdateMemberRoleMutation = { updateMemberRole: { __typename: 'Member', id: string, role: MemberRole } };

export type RemoveMemberMutationVariables = Exact<{
  spaceId: Scalars['ID']['input'];
  userId: Scalars['ID']['input'];
}>;


export type RemoveMemberMutation = { removeMember: boolean };

export type NotificationFieldsFragment = { __typename: 'Notification', id: string, type: NotificationType, text: string, image: string | null, href: string | null, read: boolean, createdAt: unknown };

export type GetNotificationsQueryVariables = Exact<{
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
}>;


export type GetNotificationsQuery = { me: { __typename: 'User', id: string, notifications: { __typename: 'NotificationsResult', unreadCount: number, items: Array<{ __typename: 'Notification', id: string, type: NotificationType, text: string, image: string | null, href: string | null, read: boolean, createdAt: unknown }> } } | null };

export type MarkNotificationReadMutationVariables = Exact<{
  id: Scalars['ID']['input'];
}>;


export type MarkNotificationReadMutation = { markNotificationRead: { __typename: 'Notification', id: string, type: NotificationType, text: string, image: string | null, href: string | null, read: boolean, createdAt: unknown } | null };

export type MarkAllNotificationsReadMutationVariables = Exact<{ [key: string]: never; }>;


export type MarkAllNotificationsReadMutation = { markAllNotificationsRead: boolean };

export type DeleteNotificationMutationVariables = Exact<{
  id: Scalars['ID']['input'];
}>;


export type DeleteNotificationMutation = { deleteNotification: boolean };

export type PostFieldsFragment = { __typename: 'Post', id: string, content: string, mediaUrls: Array<string> | null, createdAt: unknown, author: { __typename: 'User', id: string, name: string, image: string | null }, space: { __typename: 'Space', id: string, name: string, slug: string } };

export type GetUserFeedQueryVariables = Exact<{
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
}>;


export type GetUserFeedQuery = { me: { __typename: 'User', id: string, feed: Array<{ __typename: 'Post', id: string, content: string, mediaUrls: Array<string> | null, createdAt: unknown, author: { __typename: 'User', id: string, name: string, image: string | null }, space: { __typename: 'Space', id: string, name: string, slug: string } }> } | null };

export type GetSpaceFeedQueryVariables = Exact<{
  spaceId: Scalars['ID']['input'];
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
}>;


export type GetSpaceFeedQuery = { space: { __typename: 'Space', feed: Array<{ __typename: 'Post', id: string, content: string, mediaUrls: Array<string> | null, createdAt: unknown, author: { __typename: 'User', id: string, name: string, image: string | null }, space: { __typename: 'Space', id: string, name: string, slug: string } }> } | null };

export type CreatePostMutationVariables = Exact<{
  spaceId: Scalars['ID']['input'];
  content: Scalars['String']['input'];
  mediaUrls?: InputMaybe<Array<Scalars['String']['input']> | Scalars['String']['input']>;
}>;


export type CreatePostMutation = { createPost: { __typename: 'Post', id: string } };

export type DeletePostMutationVariables = Exact<{
  postId: Scalars['ID']['input'];
}>;


export type DeletePostMutation = { deletePost: boolean };

export type SpaceFieldsFragment = { __typename: 'Space', id: string, name: string, slug: string, description: string | null, cover: string, images: Array<string>, categories: Array<string>, visibility: SpaceVisibility, joinPolicy: JoinPolicy, membersCount: number | null, stripeAccountEnabled: boolean, createdAt: unknown };

export type GetAllSpacesQueryVariables = Exact<{ [key: string]: never; }>;


export type GetAllSpacesQuery = { spaces: Array<{ __typename: 'Space', id: string, name: string, slug: string, description: string | null, cover: string, images: Array<string>, categories: Array<string>, visibility: SpaceVisibility, joinPolicy: JoinPolicy, membersCount: number | null, stripeAccountEnabled: boolean, createdAt: unknown }> };

export type GetMySpacesQueryVariables = Exact<{ [key: string]: never; }>;


export type GetMySpacesQuery = { mySpaces: Array<{ __typename: 'Space', id: string, name: string, slug: string, description: string | null, cover: string, images: Array<string>, categories: Array<string>, visibility: SpaceVisibility, joinPolicy: JoinPolicy, membersCount: number | null, stripeAccountEnabled: boolean, createdAt: unknown, myMembership: { __typename: 'Member', role: MemberRole } | null }> };

export type GetSpaceQueryVariables = Exact<{
  id?: InputMaybe<Scalars['ID']['input']>;
  slug?: InputMaybe<Scalars['String']['input']>;
  membersLimit?: InputMaybe<Scalars['Int']['input']>;
}>;


export type GetSpaceQuery = { space: { __typename: 'Space', id: string, name: string, slug: string, description: string | null, cover: string, images: Array<string>, categories: Array<string>, visibility: SpaceVisibility, joinPolicy: JoinPolicy, membersCount: number | null, stripeAccountEnabled: boolean, createdAt: unknown, myMembership: { __typename: 'Member', id: string, role: MemberRole, status: MemberStatus, tier: { __typename: 'MembershipTier', id: string, name: string, price: number, interval: TierInterval } | null } | null, tiers: Array<{ __typename: 'MembershipTier', id: string, name: string, description: string | null, price: number, currency: string, interval: TierInterval, isActive: boolean, spaceId: string }>, members: Array<{ __typename: 'Member', id: string, role: MemberRole, status: MemberStatus, joinedAt: unknown, tier: { __typename: 'MembershipTier', name: string } | null, user: { __typename: 'User', id: string, name: string, email: string } }> } | null };

export type GetSpaceRecommendedEventsQueryVariables = Exact<{
  spaceId: Scalars['ID']['input'];
  limit?: InputMaybe<Scalars['Int']['input']>;
}>;


export type GetSpaceRecommendedEventsQuery = { space: { __typename: 'Space', id: string, recommendedEvents: Array<{ __typename: 'Event', id: string, title: string, location: string | null, startsAt: unknown, endsAt: unknown | null, attendeeCount: number, categories: Array<string>, price: number | null, isPaid: boolean, space: { __typename: 'Space', id: string, name: string, slug: string } | null }> } | null };

export type CreateSpaceMutationVariables = Exact<{
  input: CreateSpaceInput;
}>;


export type CreateSpaceMutation = { createSpace: { __typename: 'Space', id: string, name: string, slug: string } };

export type UpdateSpaceMutationVariables = Exact<{
  id: Scalars['ID']['input'];
  input: UpdateSpaceInput;
}>;


export type UpdateSpaceMutation = { updateSpace: { __typename: 'Space', id: string, name: string, slug: string, description: string | null, cover: string, images: Array<string>, categories: Array<string>, visibility: SpaceVisibility, joinPolicy: JoinPolicy, membersCount: number | null, stripeAccountEnabled: boolean, createdAt: unknown } };

export type DeleteSpaceMutationVariables = Exact<{
  id: Scalars['ID']['input'];
}>;


export type DeleteSpaceMutation = { deleteSpace: boolean };

export type JoinSpaceMutationVariables = Exact<{
  spaceSlug: Scalars['String']['input'];
  tierId?: InputMaybe<Scalars['ID']['input']>;
}>;


export type JoinSpaceMutation = { joinSpace: { __typename: 'Member', id: string, status: MemberStatus } };

export type LeaveSpaceMutationVariables = Exact<{
  spaceId: Scalars['ID']['input'];
}>;


export type LeaveSpaceMutation = { leaveSpace: boolean };

export type TierFieldsFragment = { __typename: 'MembershipTier', id: string, name: string, description: string | null, price: number, currency: string, interval: TierInterval, isActive: boolean, spaceId: string };

export type GetSpaceTiersQueryVariables = Exact<{
  spaceId: Scalars['ID']['input'];
}>;


export type GetSpaceTiersQuery = { space: { __typename: 'Space', id: string, tiers: Array<{ __typename: 'MembershipTier', id: string, name: string, description: string | null, price: number, currency: string, interval: TierInterval, isActive: boolean, spaceId: string }> } | null };

export type CreateTierMutationVariables = Exact<{
  spaceId: Scalars['ID']['input'];
  input: CreateTierInput;
}>;


export type CreateTierMutation = { createTier: { __typename: 'MembershipTier', id: string } };

export type ArchiveTierMutationVariables = Exact<{
  id: Scalars['ID']['input'];
}>;


export type ArchiveTierMutation = { archiveTier: boolean };

export type UserItemFieldsFragment = { __typename: 'UserItem', id: string, userId: string, type: UserItemType, promptKey: string | null, content: string, displayOrder: number, createdAt: unknown, updatedAt: unknown };

export type GetUserItemsQueryVariables = Exact<{ [key: string]: never; }>;


export type GetUserItemsQuery = { me: { __typename: 'User', id: string, userItems: Array<{ __typename: 'UserItem', id: string, userId: string, type: UserItemType, promptKey: string | null, content: string, displayOrder: number, createdAt: unknown, updatedAt: unknown }> } | null };

export type AddUserItemMutationVariables = Exact<{
  input: AddUserItemInput;
}>;


export type AddUserItemMutation = { addUserItem: { __typename: 'UserItem', id: string, userId: string, type: UserItemType, promptKey: string | null, content: string, displayOrder: number, createdAt: unknown, updatedAt: unknown } };

export type UpdateUserItemMutationVariables = Exact<{
  itemId: Scalars['ID']['input'];
  input: UpdateUserItemInput;
}>;


export type UpdateUserItemMutation = { updateUserItem: { __typename: 'UserItem', id: string, userId: string, type: UserItemType, promptKey: string | null, content: string, displayOrder: number, createdAt: unknown, updatedAt: unknown } };

export type DeleteUserItemMutationVariables = Exact<{
  itemId: Scalars['ID']['input'];
}>;


export type DeleteUserItemMutation = { deleteUserItem: boolean };

export type ReorderUserItemsMutationVariables = Exact<{
  itemIds: Array<Scalars['ID']['input']> | Scalars['ID']['input'];
}>;


export type ReorderUserItemsMutation = { reorderUserItems: Array<{ __typename: 'UserItem', id: string, userId: string, type: UserItemType, promptKey: string | null, content: string, displayOrder: number, createdAt: unknown, updatedAt: unknown }> };

export type UserFieldsFragment = { __typename: 'User', id: string, username: string | null, name: string, email: string, birthdate: string, gender: Gender | null, image: string | null, createdAt: unknown, updatedAt: unknown, sexualOrientation: Array<string>, heightCm: number | null, relationshipIntent: Array<string>, relationshipStyle: RelationshipStyle | null, hasChildren: HasChildren | null, wantsChildren: WantsChildren | null, religion: Religion | null, smoking: Smoking | null, drinking: Drinking | null, activityLevel: ActivityLevel | null, jobTitle: string | null, educationLevel: EducationLevel | null, schoolName: string | null, languages: Array<string>, ethnicity: Ethnicity | null, location: string | null };

export type GetUserQueryVariables = Exact<{
  username: Scalars['String']['input'];
}>;


export type GetUserQuery = { user: { __typename: 'User', id: string, username: string | null, name: string, email: string, birthdate: string, gender: Gender | null, image: string | null, createdAt: unknown, updatedAt: unknown, sexualOrientation: Array<string>, heightCm: number | null, relationshipIntent: Array<string>, relationshipStyle: RelationshipStyle | null, hasChildren: HasChildren | null, wantsChildren: WantsChildren | null, religion: Religion | null, smoking: Smoking | null, drinking: Drinking | null, activityLevel: ActivityLevel | null, jobTitle: string | null, educationLevel: EducationLevel | null, schoolName: string | null, languages: Array<string>, ethnicity: Ethnicity | null, location: string | null, userItems: Array<{ __typename: 'UserItem', id: string, type: UserItemType, promptKey: string | null, content: string, displayOrder: number }> } | null };

export type GetUserWithCardsQueryVariables = Exact<{
  username: Scalars['String']['input'];
}>;


export type GetUserWithCardsQuery = { user: { __typename: 'User', id: string, username: string | null, name: string, email: string, birthdate: string, gender: Gender | null, image: string | null, createdAt: unknown, updatedAt: unknown, sexualOrientation: Array<string>, heightCm: number | null, relationshipIntent: Array<string>, relationshipStyle: RelationshipStyle | null, hasChildren: HasChildren | null, wantsChildren: WantsChildren | null, religion: Religion | null, smoking: Smoking | null, drinking: Drinking | null, activityLevel: ActivityLevel | null, jobTitle: string | null, educationLevel: EducationLevel | null, schoolName: string | null, languages: Array<string>, ethnicity: Ethnicity | null, location: string | null } | null };

export type CheckUsernameQueryVariables = Exact<{
  username: Scalars['String']['input'];
}>;


export type CheckUsernameQuery = { checkUsername: boolean };

export type UpdateUserMutationVariables = Exact<{
  id: Scalars['ID']['input'];
  input: UpdateUserInput;
}>;


export type UpdateUserMutation = { updateUser: { __typename: 'User', id: string, username: string | null, name: string, email: string, birthdate: string, gender: Gender | null, image: string | null, createdAt: unknown, updatedAt: unknown, sexualOrientation: Array<string>, heightCm: number | null, relationshipIntent: Array<string>, relationshipStyle: RelationshipStyle | null, hasChildren: HasChildren | null, wantsChildren: WantsChildren | null, religion: Religion | null, smoking: Smoking | null, drinking: Drinking | null, activityLevel: ActivityLevel | null, jobTitle: string | null, educationLevel: EducationLevel | null, schoolName: string | null, languages: Array<string>, ethnicity: Ethnicity | null, location: string | null } | null };

export type UpdateLocationMutationVariables = Exact<{
  lat: Scalars['Float']['input'];
  lon: Scalars['Float']['input'];
  location?: InputMaybe<Scalars['String']['input']>;
}>;


export type UpdateLocationMutation = { updateLocation: { __typename: 'User', id: string, location: string | null, locationUpdatedAt: unknown | null, coordinates: { __typename: 'Coordinates', lat: number, lon: number } | null } };

export type GetUserRecommendedEventsQueryVariables = Exact<{
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
}>;


export type GetUserRecommendedEventsQuery = { me: { __typename: 'User', id: string, recommendedEvents: Array<{ __typename: 'Event', id: string, title: string, description: string | null, location: string | null, cover: string, startsAt: unknown, endsAt: unknown | null, maxAttendees: number | null, attendeeCount: number, myAttendeeStatus: AttendeeStatus | null, categories: Array<string>, spaceId: string, price: number | null, isPaid: boolean }> } | null };

export type GetRecommendedSpacesQueryVariables = Exact<{
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
}>;


export type GetRecommendedSpacesQuery = { me: { __typename: 'User', id: string, recommendedSpaces: Array<{ __typename: 'Space', id: string, name: string, slug: string, description: string | null, cover: string, images: Array<string>, categories: Array<string>, visibility: SpaceVisibility, joinPolicy: JoinPolicy, membersCount: number | null, stripeAccountEnabled: boolean, createdAt: unknown }> } | null };

export type GetRecommendedUsersQueryVariables = Exact<{
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
}>;


export type GetRecommendedUsersQuery = { me: { __typename: 'User', id: string, recommendedUsers: Array<{ __typename: 'User', id: string, username: string | null, name: string, image: string | null, birthdate: string, gender: Gender | null, userItems: Array<{ __typename: 'UserItem', id: string, type: UserItemType, content: string, displayOrder: number }> }> } | null };

export type GetRecommendedCategoriesWithEventsQueryVariables = Exact<{
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
}>;


export type GetRecommendedCategoriesWithEventsQuery = { me: { __typename: 'User', id: string, recommendedCategories: Array<{ __typename: 'Category', id: string, recommendedEvents: Array<{ __typename: 'Event', id: string, title: string, description: string | null, location: string | null, cover: string, startsAt: unknown, endsAt: unknown | null, maxAttendees: number | null, attendeeCount: number, myAttendeeStatus: AttendeeStatus | null, categories: Array<string>, spaceId: string, price: number | null, isPaid: boolean }> }> } | null };
